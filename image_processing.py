import numpy as np
import os
from PIL import Image
import mrcfile
import argparse
from scipy.ndimage import rotate,gaussian_filter, map_coordinates



def main():
    parser = argparse.ArgumentParser(description="Image Processing Scripts.")
    parser.add_argument('--mrc_name', type=str, required=True, help='Path to MRC images directory')
    parser.add_argument('--mrc_outname', type=str, required=True, help='Path to MRC output')
    parser.add_argument('--bin', type=int, choices=[1, 2], default=1, help='Binning factor: 1 or 2')
    parser.add_argument('--pad', type=int, choices=[0, 1], default=1, help='Padding method: 0 for zero padding, 1 for mean padding')
    
    args = parser.parse_args()
    
    mrc_array = read_image(args.mrc_name)
    
    if args.bin == 2:
        mrc_down = downsample(mrc_array,2)
    
    if args.bin == 1:
        mrc_down = mrc_array
        
    if args.pad == 0:
        mrc_down_clip = center_clip_pad0(mrc_down)
        
    elif args.pad == 1:
        mrc_down_clip = center_clip_pad_mean(mrc_down)
    #rotate_mrc = rotate_image(mrc_array,270)
    #reflect_mrc = reflect_array(mrc_array,'horizontal')
    #distortion_mrc = elastic_distortion(mrc_array)
    #bright_mrc = brightness_perturbation(mrc_array)
    write_mrc(args.mrc_outname,mrc_down_clip)
    
    

def read_image(filepath):
    """
    Read image files (supports MRC and PNG), return numpy array.
    Parameters:
        filepath (str)
    Return:
        np.ndarray
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.mrc':
        with mrcfile.open(filepath, permissive=True) as mrc:
            array = mrc.data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    return array
    
    
def write_mrc(mrc_outdir, array):
     with mrcfile.new(mrc_outdir, overwrite=True) as mrc_out:
                mrc_out.set_data(array)
                mrc_out.voxel_size = (1.0,1.0,1.0)


def downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample image by specified factor, floor the resulting dimensions.
    For example, when factor=2, each 2×2 region is averaged into one pixel.
    Parameters:
        image (np.ndarray): size (H, W)
        factor (int): scale factor of downsampling
    Return:
        np.ndarray: downsampling images
    """
    if factor <= 0:
        raise ValueError("factor must be a positive integer")
    
    if image.ndim == 2:
        h, w = image.shape
        h_crop = (h // factor) * factor
        w_crop = (w // factor) * factor
        cropped = image[:h_crop, :w_crop]
        reshaped = cropped.reshape(h_crop // factor, factor, w_crop // factor, factor)
        return reshaped.mean(axis=(1, 3))
    
    else:
        raise ValueError("Unsupported image dimensions. Input must be 2D")
        
def center_clip_pad_mean(image: np.ndarray, target_shape=(512, 512)) -> np.ndarray:
    """
    Crop and mean-pad the input image symmetrically about its center to obtain the target dimensions.

    parameters:
        image (np.ndarray): size (H, W)
        target_shape (tuple): targeted size (512, 512)
    return:
        np.ndarray: size target_shape
    """
    input_h, input_w = image.shape
    target_h, target_w = target_shape
    
    # --- x direction ---
    if input_w > target_w:
        delta_w = input_w - target_w
        left = delta_w // 2
        right = delta_w - left
        image = image[:, left:input_w - right]
    elif input_w < target_w:
        delta_w = target_w - input_w
        left = delta_w // 2
        right = delta_w - left
        pad_value = np.mean(image)
        image = np.pad(image, ((0, 0), (left, right)), mode='constant', constant_values=pad_value)
    
    # --- y direction ---
    if input_h < target_h:
        delta_h = target_h - input_h
        top = delta_h // 2
        bottom = delta_h - top
        pad_value = np.mean(image)
        image = np.pad(image, ((top, bottom), (0, 0)), mode='constant', constant_values=pad_value)
    elif input_h > target_h:
        delta_h = input_h - target_h
        top = delta_h // 2
        bottom = delta_h - top
        image = image[top:input_h - bottom, :]
    
    return image
    
def center_clip_pad0(image: np.ndarray, target_shape=(512, 512)) -> np.ndarray:
    """
    Crop and zero-pad the input image symmetrically about its center to obtain the target dimensions.

    parameters:
        image (np.ndarray): size (H, W)
        target_shape (tuple): target size (512, 512)
    return:
        np.ndarray: size target_shape
    """
    input_h, input_w = image.shape
    target_h, target_w = target_shape
    
    # --- x direction ---
    if input_w > target_w:
        delta_w = input_w - target_w
        left = delta_w // 2
        right = delta_w - left
        image = image[:, left:input_w - right]
    elif input_w < target_w:
        delta_w = target_w - input_w
        left = delta_w // 2
        right = delta_w - left
        image = np.pad(image, ((0, 0), (left, right)), mode='constant', constant_values=0)
    
    # --- y direction ---
    if input_h < target_h:
        delta_h = target_h - input_h
        top = delta_h // 2
        bottom = delta_h - top
        image = np.pad(image, ((top, bottom), (0, 0)), mode='constant', constant_values=0)
    elif input_h > target_h:
        delta_h = input_h - target_h
        top = delta_h // 2
        bottom = delta_h - top
        image = image[top:input_h - bottom, :]
    
    return image
    
def binary_array(arr: np.ndarray) -> np.ndarray:
    """
    Process each pixel value in the array according to the following rules:
    - If the value is 0, remain 0;
    - If the value exceeds 1, set to 255;
    - All other values are preserved.
    parameters：
        arr (np.ndarray): input array
    return：
        np.ndarray: processed array
    """
    result = arr.copy()
    result[result > 1] = 255
    return result
    
def rotate_image(array,angle):
    rotated_array = rotate(array, angle, reshape=False,order=0)
    
    return rotated_array
    
def reflect_array(array, axis='horizontal'):
    """
    Perform mirror reflection operation on the numpy array.
    parameters：
    - array: input numpy array（supporting 2D or 3D）
    - axis: reflecting direction
        - 'horizontal': 
        - 'vertical': 
    return：
    - the numpy array after reflecting
    """
    if axis == 'horizontal':
        return np.fliplr(array)  
    elif axis == 'vertical':
        return np.flipud(array) 
    else:
        raise ValueError("axis 应为 'horizontal', 'vertical'")
        
def elastic_distortion(image: np.ndarray, alpha: float = 25, sigma: float = 5) -> np.ndarray:
    """
    Apply elastic deformation to the 2D image.
    parameters:
    - image: input (2D numpy array)
    - alpha: Deformation strength (higher = more prominent effect)
    - sigma: Gaussian filter standard deviation (larger values yield smoother results)
    return:
    - elastic array (same size of the input array)
    """
    assert image.ndim == 2, "Supports only 2D grayscale images"

    random_state = np.random.RandomState(None)  # random seed

    # Generate random displacement field in range [-1, 1], followed by Gaussian smoothing.
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    # Grid coordinates + displacement field
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    # Bilinear interpolation sampling
    distorted = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    
    return distorted
    
def brightness_perturbation(image: np.ndarray, factor_range=(0.6, 1.4)) -> np.ndarray:
    """
    Brightness perturbation using multiplicative factor.
    parameters:
    - image: input (supporting float32)
    - factor_range: (min, max) Brightness scaling factor range, e.g.,
    return:
    - Brightness perturbation image
    """
    assert image.ndim == 2, "Supports only 2D grayscale images"
    
    factor = np.random.uniform(*factor_range)  # Sample a scaling factor randomly from the range
    img = image.astype(np.float32) * factor   
    print(factor)
    img = np.clip(img, 0.0, 1.0)  

    return img
    
if __name__ == "__main__":
    main()
