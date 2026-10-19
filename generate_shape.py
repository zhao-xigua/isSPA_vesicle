import argparse
import numpy as np
import matplotlib.pyplot as plt

def generate_ring(image_size, center, radius, thickness=7):
    """ 生成一个圆环掩码 """
    height, width = image_size
    center_x, center_y = center    # 圆心设为图像中心

    # 创建网格坐标
    y, x = np.ogrid[:height, :width]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # 确保厚度均匀地分布在 radius 的两侧
    inner_radius = radius - thickness // 2
    outer_radius = radius + thickness // 2
    
    # 创建圆环掩码
    ring_mask = (distance >= inner_radius) & (distance <= outer_radius)

    return ring_mask
    
    
def generate_elliptical_ring(image_size, center, axes, angle, thickness=7):
    """
    生成旋转椭圆环 mask

    Parameters
    ----------
    image_size : (H, W)
    center : (cx, cy)
    axes : (a, b)  # 半长轴、半短轴
    angle : float  # 角度（degree）
    thickness : int

    Returns
    -------
    mask : (H, W) bool
    """

    H, W = image_size
    cx, cy = center
    a, b = axes

    # 网格
    y, x = np.ogrid[:H, :W]

    # 平移到中心
    x_shift = x - cx
    y_shift = y - cy

    # 角度转弧度
    theta = np.deg2rad(-angle)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # 逆旋转坐标
    x_rot = x_shift * cos_t + y_shift * sin_t
    y_rot = -x_shift * sin_t + y_shift * cos_t

    # 椭圆距离（隐式函数）
    d = (x_rot**2) / (a**2) + (y_rot**2) / (b**2)

    # thickness 转换成“d空间”的宽度（近似）
    delta = thickness / max(a, b)

    inner = 1 - delta
    outer = 1 + delta

    mask = (d >= inner) & (d <= outer)

    return mask
    
    

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="生成一个带有指定半径的圆环")
    parser.add_argument("radius", type=int, help="圆环的半径（像素）")
    parser.add_argument("--size", type=int, nargs=2, default=[256, 256], help="图像大小，默认 256x256")
    parser.add_argument("--center", type=int, nargs=2, default=[128, 128], help="图像大小，默认 256x256")
    parser.add_argument("--thickness", type=int, default=7, help="圆环厚度，默认 7 像素")
    args = parser.parse_args()

    # 生成圆环
    ring_mask = generate_ring(tuple(args.size), args.center ,args.radius, args.thickness)

    # 可视化
    plt.figure(figsize=(6, 6))
    plt.imshow(ring_mask, cmap="Greens", interpolation="nearest")  # 使用绿色
    plt.title(f"Ring with Radius {args.radius}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()


            