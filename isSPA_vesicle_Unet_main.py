import os
import mrcfile
import numpy as np
from PIL import Image
#from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse
import random


# --- 1. Dataset ---
class VesicleDataset(Dataset):
    def __init__(self, mrc_dir, mask_dir, transform=None):
        self.mrc_paths = sorted([os.path.join(mrc_dir, f) for f in os.listdir(mrc_dir) if f.endswith('.mrc')])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.mrc')])
        self.transform = transform if transform is not None else lambda image, mask: {'image': image, 'mask': mask}

    def __len__(self):
        return len(self.mrc_paths)

    def __getitem__(self, idx):
        # read MRC
        with mrcfile.open(self.mrc_paths[idx], permissive=True) as mrc:
            image = mrc.data
        image = (image - image.min()) / (image.max() - image.min())
        
        # read Mask
        with mrcfile.open(self.mask_paths[idx], permissive=True) as mrc:
            mask = mrc.data
        mask = (mask / 255.0).astype(np.float32)
        
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        '''
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        '''
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        
        image = image.clone()
        mask = mask.clone()
        #print(image.shape)
        
        #print("Image shape:", image.shape)  #  torch.Size([1, 512, 512])
        #print("Mask shape:", mask.shape)    #  torch.Size([1, 512, 512])
        #print("Unique values in mask:", np.unique(augmented['mask']))

        return image, mask
        
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetFull(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetFull, self).__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)

class CombinedLoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        preds = torch.sigmoid(logits)  
        d_loss = dice_loss(preds, targets)
        return bce_loss + d_loss


# --- 3. Dice coefficient and validation Function ---
def dice_coefficient(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection + 1e-8) / (union + 1e-8)
    return dice.mean().item()


def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0
    total_dice = 0
    #pos_weight = torch.tensor([4.0], device=device)
    #criterion = CombinedLoss(pos_weight=pos_weight)
    with torch.no_grad():
        for image, mask in dataloader:
            image, mask = image.to(device), mask.to(device)
            logits = model(image)
            pred = torch.sigmoid(logits)
            loss = criterion(logits, mask)
            dice = dice_coefficient(pred, mask)
            total_loss += loss.item()
            total_dice += dice
    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    
    return avg_loss, avg_dice


# --- 4. Train ---
def train(model, train_loader, val_loader, device, pos_weight, epochs):
    model.to(device)
    criterion = CombinedLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    train_losses = []
    val_losses = []
    val_dices = []

    # Early stop parameters
    best_dice = -1
    patience = 100
    min_delta = 0.001
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for image, mask in train_loader:
            image, mask = image.to(device), mask.to(device)
            logits = model(image)
            loss = criterion(logits, mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        # Evaluate on validation set
        val_loss, val_dice = evaluate(model, val_loader, device, criterion)
        
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
        
        if val_dice > best_dice + min_delta:
            best_dice = val_dice
            patience_counter = 0
            #model_name = f"unet_model_batch{batch_size}_dice{val_dice:.4f}.pth"
            torch.save(model.state_dict(), "unet_model.pth")
            
            # save model information
            with open("best_model_info.txt", "w") as f:
                f.write(f"Best Epoch: {epoch+1}\n")
                f.write(f"Best Val Dice: {val_dice:.6f}\n")

            print(f"New best model saved (Dice={val_dice:.4f})")
            
            
        else:
            patience_counter += 1
            print(f"No improvement. Patience {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    #torch.save(model.state_dict(), "unet_model.pth")
    
    
    # ---------- drawing curves ----------
    #epochs_range = range(1, epochs + 1)
    epochs_range = range(1, len(train_losses)+1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, label="Train Loss", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train Loss")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, val_losses, label="Val Loss", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, val_dices, label="Val Dice", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Dice Coefficient")
    plt.title("Validation Dice")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()
    

# --- 5. Predictoion and save ---
def predict_and_save(model, dataloader, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, (image, _) in enumerate(dataloader):
            image = image.to(device)
            pred = torch.sigmoid(model(image))
            pred_mask = (pred > 0.5).float().squeeze().cpu().numpy()
            
            with mrcfile.new(os.path.join(output_dir, f"pred_{i:03d}.mrc"), overwrite=True) as mrc:
                mrc.set_data(pred_mask.astype(np.float32))
            #result = Image.fromarray(pred_mask.astype(np.uint8))
            #result.save(os.path.join(output_dir, f"pred_{i:03d}.png"))
            
def predict_on_new_images(model, predict_dir, output_dir, device):
    model.eval()

    
    mrc_paths = sorted([os.path.join(predict_dir, f) for f in os.listdir(predict_dir) if f.endswith('.mrc')])

    os.makedirs(output_dir, exist_ok=True)

    
    predict_transform = A.Compose([
        A.Normalize(mean=(0.0,), std=(1.0,)),
        ToTensorV2(),
    ])

    with torch.no_grad():
        for i, mrc_path in enumerate(mrc_paths):
            
            with mrcfile.open(mrc_path, permissive=True) as mrc:
                image = mrc.data
            image = (image - image.min()) / (image.max() - image.min())

            augmented = predict_transform(image=image)
            image_tensor = augmented['image'].unsqueeze(0).to(device)  # add batch dimension

            # prediction
            pred = model(image_tensor)
            pred_mask = (torch.sigmoid(pred) > 0.5).float().squeeze().cpu().numpy()

            # save results
            output_path = os.path.join(output_dir, os.path.basename(mrc_path))
            with mrcfile.new(output_path, overwrite=True) as mrc:
                mrc.set_data(pred_mask.astype(np.float32))
                
            print(f"Saved prediction: {output_path}")

# --- 6. Main  ---
def main():
    parser = argparse.ArgumentParser(description="Train UNet on MRC and masks")
    parser.add_argument('--mrc_dir', type=str, required=True, help='Path to MRC images directory')
    parser.add_argument('--mask_dir', type=str, required=True, help='Path to MRC mask directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save prediction results')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Training batch size')
    parser.add_argument('--predict', action='store_true', help='Run prediction on new images')
    parser.add_argument('--predict_dir', type=str, help='Path to MRC images for prediction')
    parser.add_argument('--model_path', type=str, default="unet_model.pth", help='Path to the trained model')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    args = parser.parse_args()

    mrc_dir = args.mrc_dir
    mask_dir = args.mask_dir
    output_dir = args.output_dir
    epochs = args.epochs
    batch_size = args.batch_size
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.predict:
        # only predict, dont train
        assert args.predict_dir is not None, "You must specify --predict_dir when using --predict."
        model = UNetFull()
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        predict_on_new_images(
            model=model,
            predict_dir=args.predict_dir,
            output_dir=args.output_dir,
            device=device
        )
    else:
        # training pipeline
        assert args.mrc_dir is not None and args.mask_dir is not None, "You must specify --mrc_dir and --mask_dir when training."
        
        # data augmentation
        train_transform = A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        A.Normalize(mean=(0.0,), std=(1.0,)),  
        ToTensorV2(),
        ])
        
        val_transform = A.Compose([
        A.Normalize(mean=(0.0,), std=(1.0,)),
        ToTensorV2(),
        ])
    
    
    
        # data loader
        dataset = VesicleDataset(mrc_dir, mask_dir)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_indices, val_indices = torch.utils.data.random_split(range(len(dataset)), [train_size, val_size])

        train_dataset = torch.utils.data.Subset(VesicleDataset(mrc_dir, mask_dir, transform=train_transform), train_indices)
        val_dataset = torch.utils.data.Subset(VesicleDataset(mrc_dir, mask_dir, transform=val_transform), val_indices)
        print(f"Total training samples: {len(train_dataset)}")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        print("Validation image files:")
        for i in val_dataset.indices:
            print(os.path.basename(val_dataset.dataset.mrc_paths[i]))
        
        #  pos_weight
        '''
        total_pos = 0
        total_neg = 0
        for _, mask in DataLoader(dataset, batch_size=1):
            total_pos += mask.sum().item()
            total_neg += (mask == 0).sum().item()
        pos_weight = total_neg / (total_pos + 1e-6)
        '''
        pos_weight_tensor = torch.tensor([4], device=device)
        
        # model train and predict
        model = UNetFull()
        train(model, train_loader, val_loader, device, epochs=epochs, pos_weight=pos_weight_tensor)
        model.load_state_dict(torch.load("unet_model.pth"))
        predict_and_save(model, val_loader, device, output_dir)

if __name__ == "__main__":
    main()
