import numpy as np
import cv2
import mrcfile
import argparse
from image_processing import write_mrc, center_clip_pad0


def main():
    # 1. 命令行参数解析
    parser = argparse.ArgumentParser(description="According to mask.txt generate filled mask image.")
    parser.add_argument('--mask_txt', type=str, required=True, help='Path to mask txt directory')
    parser.add_argument('--filled_mask_mrc', type=str, required=True, help='Path to filled mask output')
    parser.add_argument('--scale', type=float, default=2, help='Default=2; center/2')
    parser.add_argument('--thick', type=float, default=2, help='Default=2; radius for your mask.txt file; radius+(thick/2)')
    parser.add_argument('--image_size', type=str, default='576,409', help='width,height(default: 576,409)'
)
    
    args = parser.parse_args()
    
    width, height = map(int, args.image_size.split(','))
    
    circle_list = read_and_process_file(args.mask_txt, args.scale)
    circle_array = draw_elliptical_disks(circle_list, height, width, args.thick)
    #print(circle_array.shape)
    #clip_array = center_clip_pad0(circle_array)
    write_mrc(args.filled_mask_mrc, circle_array)


def read_and_process_file(file_path, scale):
    circles = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                
                center_str = parts[2].split('=')[1]  
                radius_str = parts[3].split('=')[1]  
                angle_str = parts[4].split('=')[1]   
                
                center_x, center_y = map(float, center_str.split(','))
                radius_long, radius_short = map(float, radius_str.split(','))
                angle = float(angle_str)
                
                center_x /= scale
                center_y /= scale
                radius_long /= scale
                radius_short /= scale

                circles.append(((center_x, center_y), (radius_long, radius_short), angle))
    
    return circles



def draw_elliptical_disks(ellipse_list, image_size_x, image_size_y, thick=2):
    
    img = np.zeros((image_size_x, image_size_y), dtype=np.float32)
    
    for center, axes, angle in ellipse_list:
    
        cx , cy = center
        a, b = axes
        angle = -1 * angle
        
        a = a + thick / 2
        b = b + thick / 2
        
        center_int = (int(cx), int(cy))
        axes_int = (int(a), int(b))
        
        #radius_pix = radius_pix + (thick/2)//2   #因为圆环的厚度在ms5的图中是5pix，因此在ms10图中就是5/2，又因为(5/2)是分别加在圆环两侧，故(5/2)//2
        
        #cv2.circle(img, center, int(radius_pix), color=255, thickness=-1)   # cv2.circle绘图默认坐标原点在左上角
        
        cv2.ellipse(img, center_int, axes_int, angle, 0, 360, color=255, thickness=-1)
        
    return img
    

if __name__ == "__main__":
    main()
