import numpy as np
import healpy as hp
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import argparse
import math, mrcfile


def main():
    parser = argparse.ArgumentParser(description="Sphere → Ellipsoid sampling pipeline")
    parser.add_argument("--axes", nargs=3, type=float, required=True, help="ellipsoid axes: a b c")
    parser.add_argument("--center", nargs=3, type=float, required=True, help="ellipsoid center: x y z")
    parser.add_argument("--angle", type=float, required=True, help="ellipsoid angle")
    parser.add_argument("--min_dist", type=float, default=20, help="minimum distance for density filtering")
    parser.add_argument("--output", type=str, default="ellipsoid.png", help="output png path")
    parser.add_argument('--protein_orient',type=int, choices=[1, -1], help='1: protein oriented toward vesicle exterior (outward-facing); -1: protein oriented toward vesicle interior (inward-facing)')
    parser.add_argument("--elev", type=float, default=30)
    parser.add_argument("--azim", type=float, default=45)

    args = parser.parse_args()
    
    anticipating_nside = auto_sampling_density(args.axes, args.min_dist)
    
    sphere_pts, normals = generate_sphere_points(anticipating_nside)

    q_global, q_normals = map_points_to_ellipsoid(sphere_pts, axes=args.axes, ellipsoid_center=args.center, angle_deg=args.angle, protein_orient=args.protein_orient)
    
    mask = points3d_to_2d_mask(q_global, q_normals)

    with mrcfile.new("mask.mrc", overwrite=True) as mrc:
        mrc.set_data(mask)
    
    plot_3d_points(q_global, output_path=args.output, elev=args.elev, azim=args.azim)



def auto_sampling_density(axes, min_dist):

    # Start the calculation from nside = 32
    
    nside = 32
    N = 12 * nside * nside
    
    long_a, short_b = axes[0], axes[1]
    
    sphere_spacing = short_b * math.sqrt(4 * math.pi / N)
    
    axis_ratio = long_a / short_b
    
    ellipsoid_spacing = axis_ratio * sphere_spacing
    
    criterion = ellipsoid_spacing / min_dist
    
    ### If ellipsoid_spacing / min_dist is greater than 1.2, the sampling is too coarse and should be refined (nside > 32)
    ### If ellipsoid_spacing / min_dist is between 0.8 and 1.2, the sampling resolution is appropriate (nside = 32)
    ### If ellipsoid_spacing / min_dist is less than 0.8, the sampling is too dense and should be made sparser (nside < 32)
    
    if criterion > 1.2 or criterion < 0.8: 
        times_N = criterion * criterion
        anticipating_N = times_N * N
        anticipating_nside =  math.ceil(np.sqrt(anticipating_N / 12))
        
    if criterion <= 1.2 and criterion >= 0.8: 
        anticipating_nside = nside
        
    return anticipating_nside
        


def generate_sphere_points(nside=32):
    """ Generate spherical sampling points and normals using HEALPix """
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    
    # unit sphere: 
    x0 = 0 
    y0 = 0 
    z0 = 0
    r = 1
    
    x = x0 + r * np.sin(theta) * np.cos(phi)
    y = y0 + r * np.sin(theta) * np.sin(phi)
    z = z0 + r * np.cos(theta)
    
    # normal vectors (normalization)
    nx = (x - x0) / r
    ny = (y - y0) / r
    nz = (z - z0) / r
    
    coords = np.stack([x, y, z], axis=1)
    normals = np.stack([nx, ny, nz], axis=1)
    
    return coords, normals
    


def map_points_to_ellipsoid(points, axes, ellipsoid_center, angle_deg=0.0, protein_orient=1):
    """
    unit sphere → ellipsoid（origin）→ translate to target coordinate

    Parameters
    ----------
    points : (N,3)
        sampling points on unit sphere（origin）
    axes : (a,b,c)
        ellipsoidal axes length
    ellipsoid_center : (3,)

    Returns
    -------
    mapped_points : (N,3)
    """

    a, b, c = axes
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # --- Step 1: scale factor caculation ---
    scale = 1.0 / np.sqrt((x**2)/(a**2) + (y**2)/(b**2) + (z**2)/(c**2))

    # --- Step 2: map points to ellipsoid（origin） ---
    q = points * scale[:, None]

    
    # --- Step 3: caculation normal vectors
    nx = q[:, 0] / (a**2)
    ny = q[:, 1] / (b**2)
    nz = q[:, 2] / (c**2)

    q_normals = np.stack([nx, ny, nz], axis=1)
    
    q_normals = q_normals * protein_orient
    
    # normalization
    q_normals /= np.linalg.norm(q_normals, axis=1, keepdims=True)
    
    # --- Step 4: rotate around the Z-axis ---
    if angle_deg != 0.0:
        theta = np.deg2rad(angle_deg)

        Rz = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])

        q = q @ Rz.T
        q_normals = q_normals @ Rz.T

    # --- Step 5: translation ---
    q_global = q + np.array(ellipsoid_center)
    

    return q_global, q_normals
    

    
def plot_3d_points(points, output_path, elev=30, azim=45, point_size=5, color='blue', equal_aspect=True):
    """
    plotting 3D point cloud to PNG img

    Parameters
    ----------
    points : (N,3) array
    output_path : str
        output path of png
    elev : float
        elevation angle (default = 30)
    azim : float
        azimuth angle(default = 45)
    point_size : float
    color : str or array
    equal_aspect : bool
    """

    points = np.asarray(points)
    x, y, z = points[:,0], points[:,1], points[:,2]

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, s=point_size, c=color)

    # setting viewing angle
    ax.view_init(elev=elev, azim=azim)

    # keep XYZ aspect ratio ?
    if equal_aspect:
        max_range = np.array([
            x.max()-x.min(),
            y.max()-y.min(),
            z.max()-z.min()
        ]).max() / 2.0

        mid_x = (x.max()+x.min()) / 2
        mid_y = (y.max()+y.min()) / 2
        mid_z = (z.max()+z.min()) / 2

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    

def points3d_to_2d_mask(points, normals, shape=(512, 512)):
    """
    project 3D coordinates to 2D mask (only keep x and y)

    Parameters
    ----------
    points : (N,3) ndarray
    shape : tuple
        (H, W)

    Returns
    -------
    mask : (H,W) float32
        binary image
    """

    H, W = shape
    mask = np.zeros((H, W), dtype=np.float32)
    
    # only keep x and y direction
    nx = normals[:, 0]
    ny = normals[:, 1]
    
    # --- Step 1: extract x, y coord---
    x0 = np.round(points[:, 0]).astype(int)
    y0 = np.round(points[:, 1]).astype(int)

    length = 10.0
    x1 = np.round(points[:, 0] + length * nx).astype(int)
    y1 = np.round(points[:, 1] + length * ny).astype(int)

    # --- Step 3: edge protection ---
    def valid(x, y):
        return (x >= 0) & (x < W) & (y >= 0) & (y < H)
    
    v0 = valid(x0, y0)
    mask[y0[v0], x0[v0]] = 1.0
    
    # --- Step 2: draw line (Bresenham-like sampling via interpolation) ---
    steps = 20  # Line sampling density
    for i in range(len(points)):
        if not v0[i]:
            continue

        xs = np.linspace(x0[i], x1[i], steps)
        ys = np.linspace(y0[i], y1[i], steps)

        xi = np.round(xs).astype(int)
        yi = np.round(ys).astype(int)

        v = valid(xi, yi)
        mask[yi[v], xi[v]] = 0.7  

    # --- Step 3: endpoint highlight ---
    v1 = valid(x1, y1)
    for i in range(len(x1)):
        if not v1[i]:
            continue

        cx, cy = x1[i], y1[i]

        # 3x3 bold point
        mask[cy-1:cy+1, cx-1:cx+1] = 1.0
        

    return mask


if __name__ == "__main__":
    main()