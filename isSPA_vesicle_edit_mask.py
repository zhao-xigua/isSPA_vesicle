#!/usr/bin/env python3
import argparse
import numpy as np
import mrcfile


def main():
    args = parse_args()

    with mrcfile.open(args.input, permissive=True) as mrc:
        data = mrc.data.copy()

    if data.shape != (512, 512):
        raise ValueError(f"Input MRC must be 512x512, got {data.shape}")

    mask = generate_center_mask(data.shape, args.size)

    result = data * mask

    with mrcfile.new(args.output, overwrite=True) as mrc_out:
        mrc_out.set_data(result.astype(np.float32))

    print("Done.")
    print(f"Input : {args.input}")
    print(f"Output: {args.output}")
    print(f"Mask size: {args.size}")


def parse_args():
    parser = argparse.ArgumentParser(description="Apply center square mask to a 512x512 MRC image")
    
    parser.add_argument("-i", "--input", required=True, help="Input MRC file (512x512)")
    parser.add_argument("-o", "--output", required=True, help="Output MRC file")
    parser.add_argument("-s", "--size", type=int, nargs=2, required=True, metavar=('HEIGHT', 'WIDTH'), help="Center rectangle size (e.g., 409 409)")
    
    return parser.parse_args()


def generate_center_mask(shape, size_hw):
    H, W = shape
    h, w = size_hw

    mask = np.zeros((H, W), dtype=np.float32)

    cy, cx = H // 2, W // 2


    hy = h // 2
    if h % 2 == 0:
        y1, y2 = cy - hy, cy + hy
    else:
        y1, y2 = cy - hy, cy + hy + 1

    
    hx = w // 2
    if w % 2 == 0:
        x1, x2 = cx - hx, cx + hx
    else:
        x1, x2 = cx - hx, cx + hx + 1

    
    y1, y2 = max(0, y1), min(H, y2)
    x1, x2 = max(0, x1), min(W, x2)

    mask[y1:y2, x1:x2] = 1.0

    return mask


if __name__ == "__main__":
    main()