import numpy as np
import mrcfile
import argparse


SHELL_WIDTH = 4
DENSITY = 1.0


def generate_cylinder_mrc(filename,shape,voxel_size,radius,z_range):
    zdim, ydim, xdim = shape
    data = np.zeros(shape, dtype=np.float32)

    x0, y0 = xdim // 2, ydim // 2
    z_min, z_max = z_range

    inner_radius = radius - SHELL_WIDTH // 2
    outer_radius = radius + SHELL_WIDTH // 2

    for z in range(z_min, z_max):
        for y in range(ydim):
            for x in range(xdim):
                r2 = (x - x0) ** 2 + (y - y0) ** 2
                if inner_radius ** 2 <= r2 <= outer_radius ** 2:
                    data[z, y, x] = DENSITY

    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = voxel_size
        mrc.update_header_from_data()

        mrc.header.cella.x = voxel_size * xdim
        mrc.header.cella.y = voxel_size * ydim
        mrc.header.cella.z = voxel_size * zdim

        mrc.header.mx = xdim
        mrc.header.my = ydim
        mrc.header.mz = zdim

        mrc.update_header_stats()

    print(f"MRC file '{filename}' generated successfully!")


def calculate_distance_from_mem(shape, z_range):
    zdim, ydim, xdim = shape
    z_min, z_max = z_range

    z_center = zdim / 2.0

    dist_to_z_min = abs(z_center - z_min)
    dist_to_z_max = abs(z_center - z_max)

    distance_from_mem = min(dist_to_z_min, dist_to_z_max)

    print(f"[When map transmembrane region aligns with cylinder] distance_from_mem = {distance_from_mem}")

    return distance_from_mem


def parse_args():
    parser = argparse.ArgumentParser(description="Generate cylindrical MRC volume")

    parser.add_argument("--filename", type=str, required=True)

    parser.add_argument("--shape", type=int, nargs=3, default=[200, 200, 200], help="Volume shape: z y x (default: %(default)s)")

    parser.add_argument("--voxel_size", type=float, default=1.0, help="Voxel size (default: %(default)s)")

    parser.add_argument("--radius", type=float, default=30.0, help="Cylinder radius (default: %(default)s)")

    parser.add_argument("--z_range", type=int, nargs=2, default=[30, 70], help="Cylinder z range: z_min z_max (default: %(default)s)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    distance_from_mem = calculate_distance_from_mem(shape=tuple(args.shape), z_range=tuple(args.z_range))
    
    generate_cylinder_mrc(
        filename=args.filename,
        shape=tuple(args.shape),
        voxel_size=args.voxel_size,
        radius=args.radius,
        z_range=tuple(args.z_range)
    )