import numpy as np
import os
from PIL import Image
import mrcfile
import argparse
from scipy.ndimage import rotate,gaussian_filter, map_coordinates



def main():
    # 1. 命令行参数解析
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
    读取图像文件（支持 MRC 和 PNG），返回 numpy 数组格式。
    参数:
        filepath (str): 图像文件路径（.mrc）
    返回:
        np.ndarray: 图像的 NumPy 数组
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
    将图像按给定倍数进行降采样（向下取整处理尺寸）。
    例如 factor=2 时，2x2 区域平均为 1 个像素。
    参数:
        image (np.ndarray): 输入图像，形状为 (H, W)
        factor (int): 降采样倍数
    返回:
        np.ndarray: 降采样后的图像
    """
    if factor <= 0:
        raise ValueError("factor 必须为正整数")
    
    if image.ndim == 2:
        # 灰度图像
        h, w = image.shape
        h_crop = (h // factor) * factor
        w_crop = (w // factor) * factor
        cropped = image[:h_crop, :w_crop]
        reshaped = cropped.reshape(h_crop // factor, factor, w_crop // factor, factor)
        return reshaped.mean(axis=(1, 3))
    
    else:
        raise ValueError("不支持的图像维度，必须是 2D")
        
def center_clip_pad_mean(image: np.ndarray, target_shape=(512, 512)) -> np.ndarray:
    """
    将输入图像沿中心对称裁剪和补零，得到目标尺寸。
    对于过大维度：从中心对称裁剪；
    对于过小维度：从中心对称补零。
    参数:
        image (np.ndarray): 输入图像，形状为 (H, W)
        target_shape (tuple): 目标尺寸，例如 (512, 512)
    返回:
        np.ndarray: 处理后的图像，形状为 target_shape
    """
    input_h, input_w = image.shape
    target_h, target_w = target_shape
    
    # --- 水平方向（x方向）裁剪 ---
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
    
    # --- 垂直方向（y方向）补零 ---
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
    将输入图像沿中心对称裁剪和补零，得到目标尺寸。
    对于过大维度：从中心对称裁剪；
    对于过小维度：从中心对称补零。
    参数:
        image (np.ndarray): 输入图像，形状为 (H, W)
        target_shape (tuple): 目标尺寸，例如 (512, 512)
    返回:
        np.ndarray: 处理后的图像，形状为 target_shape
    """
    input_h, input_w = image.shape
    target_h, target_w = target_shape
    
    # --- 水平方向（x方向）裁剪 ---
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
    
    # --- 垂直方向（y方向）补零 ---
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
    将数组中每个像素值按照以下规则处理：
    - 如果值为 0，仍为 0；
    - 如果值 > 1，设为 255；
    - 其它值保持不变。
    参数：
        arr (np.ndarray): 输入的数组
    返回：
        np.ndarray: 处理后的数组
    """
    result = arr.copy()
    result[result > 1] = 255
    return result
    
def rotate_image(array,angle):
    rotated_array = rotate(array, angle, reshape=False,order=0)
    
    return rotated_array
    
def reflect_array(array, axis='horizontal'):
    """
    对 numpy array 进行镜像反射操作。
    参数：
    - array: 输入的 numpy 数组（支持 2D 或 3D）
    - axis: 反射方向
        - 'horizontal': 左右翻转（沿 y 轴）
        - 'vertical': 上下翻转（沿 x 轴）
    返回：
    - 反射后的 numpy 数组
    """
    if axis == 'horizontal':
        return np.fliplr(array)  # 左右反转
    elif axis == 'vertical':
        return np.flipud(array)  # 上下反转
    else:
        raise ValueError("axis 应为 'horizontal', 'vertical'")
        
def elastic_distortion(image: np.ndarray, alpha: float = 25, sigma: float = 5) -> np.ndarray:
    """
    对2D图像进行弹性形变。
    参数:
    - image: 输入图像（2D numpy array）
    - alpha: 形变强度（越大越明显）
    - sigma: 高斯滤波标准差（越大越平滑）
    返回:
    - 形变后的图像（与原图同形状）
    """
    assert image.ndim == 2, "仅支持 2D 图像（灰度）"

    random_state = np.random.RandomState(None)  # 随机数种子（每次不一样）

    # 生成随机位移场（[-1, 1]之间），再用高斯滤波平滑
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    # 网格坐标 + 位移场
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    # 双线性插值采样
    distorted = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    
    return distorted
    
def brightness_perturbation(image: np.ndarray, factor_range=(0.6, 1.4)) -> np.ndarray:
    """
    对图像进行亮度扰动（乘法因子方式）
    参数:
    - image: 输入图像（支持 float32）
    - factor_range: (min, max) 亮度缩放因子范围，例如 (0.6, 1.4)
    返回:
    - 亮度扰动后的图像
    """
    assert image.ndim == 2, "仅支持 2D 图像（灰度）"
    
    factor = np.random.uniform(*factor_range)  # 从范围中随机取一个缩放因子
    img = image.astype(np.float32) * factor   # 扩展亮度
    print(factor)
    img = np.clip(img, 0.0, 1.0)  # 如果是float图像，确保归一化范围内

    return img
    
if __name__ == "__main__":
    main()
