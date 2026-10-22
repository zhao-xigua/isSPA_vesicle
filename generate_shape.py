import argparse
import numpy as np
import matplotlib.pyplot as plt

def generate_ring(image_size, center, radius, thickness=7):
    """ generate a ring mask """
    height, width = image_size
    center_x, center_y = center    

    y, x = np.ogrid[:height, :width]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    inner_radius = radius - thickness // 2
    outer_radius = radius + thickness // 2
    
    ring_mask = (distance >= inner_radius) & (distance <= outer_radius)

    return ring_mask
    
    
def generate_elliptical_ring(image_size, center, axes, angle, thickness=7):
    """
    Generate rotated elliptical ring mask

    Parameters
    ----------
    image_size : (H, W)
    center : (cx, cy)
    axes : (a, b)  # semi-major axis, semi-minor axis
    angle : float  # angle（degree）
    thickness : int

    Returns
    -------
    mask : (H, W) bool
    """

    H, W = image_size
    cx, cy = center
    a, b = axes

    y, x = np.ogrid[:H, :W]

    x_shift = x - cx
    y_shift = y - cy

    theta = np.deg2rad(-angle)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x_rot = x_shift * cos_t + y_shift * sin_t
    y_rot = -x_shift * sin_t + y_shift * cos_t

    d = (x_rot**2) / (a**2) + (y_rot**2) / (b**2)

    delta = thickness / max(a, b)

    inner = 1 - delta
    outer = 1 + delta

    mask = (d >= inner) & (d <= outer)

    return mask
    
    

def main():
    parser = argparse.ArgumentParser(description="Generate a ring mask with specified radius")
    parser.add_argument("radius", type=int, help="Radius of the ring (pixels)")
    parser.add_argument("--size", type=int, nargs=2, default=[256, 256], help="Image size. Default: 256x256")
    parser.add_argument("--center", type=int, nargs=2, default=[128, 128], help="Image center. Default: 128x128")
    parser.add_argument("--thickness", type=int, default=7, help="Thickness of the ring mask. Default: 7 pixels")
    args = parser.parse_args()

    ring_mask = generate_ring(tuple(args.size), args.center ,args.radius, args.thickness)

    plt.figure(figsize=(6, 6))
    plt.imshow(ring_mask, cmap="Greens", interpolation="nearest")  
    plt.title(f"Ring with Radius {args.radius}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()


            