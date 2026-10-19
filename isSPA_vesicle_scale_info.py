import argparse
import re

def process_line(line, clip_w, clip_h, out_w, out_h):
    # 匹配基本元素
    pattern = r'^(.*?)\s+image_size=(\d+),(\d+)\s+center=(\d+\.?\d*),(\d+\.?\d*)\s+radius=(\d+\.?\d*),(\d+\.?\d*)\s+angle=(\d+\.?\d*)\s+axis_ratio=(\d+\.?\d*)\s+type=(\w+)$'
    match = re.match(pattern, line.strip())
    
    if not match:
        print(f"Warning: line does not match expected format: {line}")
        return None
    
    filename, width, height, center_x, center_y, radius_x, radius_y, angle, ratio, types = match.groups()
    
    if int(width) != 512 or int(height) != 512:
        raise ValueError(f"Invalid image size {width}x{height}. This script only supports 512x512 coordinate conversion.")
    
    center_x = float(center_x)
    center_y = float(center_y)
    radius_x = float(radius_x)
    radius_y = float(radius_y)
    
    # 计算变换
    deltax = (int(width) - clip_w) / 2
    deltay = (int(height) - clip_h) / 2
    
    margin_x = out_w - clip_w * 10
    margin_y = out_h - clip_h * 10
    
    # NOTE: empirical scaling factor based on ms10->micrograph mapping
    new_center_x = (center_x - deltax) * 10 + margin_x     
    new_center_y = (center_y - deltay) * 10 + margin_y    # 409直接乘10是4090，可是当时clip时沿着中心clip，两边各自丢掉了3个像素，所以+3
    new_radius_x = radius_x * 10
    new_radius_y = radius_y * 10
    
    # 新的image_size
    new_image_size = f"image_size={out_w},{out_h}"
    
    # 重新组合成一行
    new_line = f"{filename}\t{new_image_size}\tcenter={new_center_x:.2f},{new_center_y:.2f}\tradius={new_radius_x:.2f},{new_radius_y:.2f}\tangle={angle}\taxis_ratio={ratio}\ttype={types}"
    
    return new_line

def main():
    parser = argparse.ArgumentParser(description="处理中心坐标和半径并输出新文件")
    parser.add_argument('input_file', help="输入文件路径")
    parser.add_argument('output_file', help="输出文件路径")
    
    parser.add_argument('--ms10_mic_size', type=int, nargs=2, required=True, metavar=('clip_w', 'clip_h'), help="ms10 size, e.g. --ms10_size 409 409")
    parser.add_argument('--micrograph_size', type=int, nargs=2, required=True, metavar=('out_w', 'out_h'), help="micrograph size, e.g. --micrograph_size 4096 4096")
    
    args = parser.parse_args()
    
    clip_w, clip_h = args.ms10_mic_size
    out_w, out_h = args.micrograph_size
    
    with open(args.input_file, 'r') as f:
        lines = f.readlines()
    
    processed_lines = []
    for line in lines:
        if line.strip():  # 跳过空行
            new_line = process_line(line, clip_w, clip_h, out_w, out_h)
            if new_line:
                processed_lines.append(new_line)
    
    with open(args.output_file, 'w') as f:
        for line in processed_lines:
            f.write(line + '\n')
    
    print(f"处理完成！结果保存到: {args.output_file}")

if __name__ == "__main__":
    main()


