#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import cv2
import mrcfile
from scipy.ndimage import maximum_filter, label
from scipy.spatial.distance import cdist


def main():
    parser = argparse.ArgumentParser(description="Fit ellipses from mask.mrc")

    parser.add_argument("--input", "-i", required=True,help="input mask.mrc")
    parser.add_argument("--output_txt", "-ot", required=True,help="output txt file")
    parser.add_argument("--output_mrc", "-om", required=True,help="output fitted ellipse mask mrc")
    parser.add_argument("--min_area", type=float, default=100,help="minimum area threshold")
    parser.add_argument("--peak_size", type=int, default=18, help="window size for local maximum detection (default: 18)")
    parser.add_argument("--min_distance", type=float, default=8, help="minimum distance between neighboring peaks (default: 8)")

    args = parser.parse_args()

    with mrcfile.open(args.input, permissive=True) as mrc:
        mask = mrc.data.copy()
    
    mask = np.squeeze(mask)
    h, w = mask.shape

    results = fit_ellipses_from_mask(mask, args.min_area, peak_size=args.peak_size, min_distance=args.min_distance)

    with open(args.output_txt, "w") as f:

        for r in results:
            cx, cy = r["center"]
            major_r = r["major_axis"]/2
            minor_r = r["minor_axis"]/2
            angle = r["angle"]
            axis_ratio = r["axis_ratio"]
            shape_type = r["type"]
            
            line = (
                f"{args.input}\t"
                f"image_size={w},{h}\t"
                f"center={cx:.2f},{cy:.2f}\t"
                f"radius={major_r:.2f},{minor_r:.2f}\t"
                f"angle={angle:.2f}\t"
                f"axis_ratio={axis_ratio:.2f}\t"
                f"type={shape_type}\n"
            )
            
            f.write(line)

    fitted_mask = draw_ellipses_mask((h, w), results, expand=0.5)

    with mrcfile.new(args.output_mrc, overwrite=True) as mrc:
        mrc.set_data(fitted_mask.astype(np.float32))

    print(f"[INFO] Done. Found {len(results)} objects.")


def fit_ellipses_from_mask(mask, min_area=100, peak_size=18, min_distance=8):
    
    raw_mask = (mask > 0).astype(np.uint8)
    
    h, w = raw_mask.shape
    
    dist_transform = cv2.distanceTransform(raw_mask,cv2.DIST_L2,5)
    
    local_max = (dist_transform ==maximum_filter(dist_transform,size=peak_size))

    local_max = local_max & (dist_transform > 0)
    
    local_max = remove_close_peaks(local_max, dist_transform, min_distance=min_distance)
    
    # Label each marker
    peak_label, num_peaks = label(local_max)
    
    if num_peaks == 0:
        return []
    
    ##################################################
    # 3. connected component + peak assignment
    ##################################################
    
    
    mask_label, num_components = label(raw_mask)
    
    results = []
    
    # Iterate over each connected component in the original mask
    for comp_id in range(1, num_components + 1):
        
        # Current connected component
        component = (mask_label == comp_id)
        
        # Find distance peaks within the current component
        peak_coords = np.argwhere(
            (peak_label > 0) &
            (mask_label == comp_id)
        )

        
        # no peak
        # skip this component
        if len(peak_coords)==0:
            continue
        
        
        # Case 1：
        # Component with a single peak
        
        if len(peak_coords)==1:
        
            process_component(component, results, min_area)
            
        # Case 2：
        # Component with multiple peaks
        
        else:
        
            coords = np.argwhere(component)

            # Compute distances from each pixel to all peaks

            dist = cdist(coords, peak_coords)

            nearest = np.argmin(dist, axis=1)
            
            # Split the component based on the nearest peak

            for peak_id in range(len(peak_coords)):

                selected_coords = coords[nearest == peak_id]

                if len(selected_coords)<5:
                    continue

                sub_component=np.zeros_like(raw_mask)
                
                sub_component[ selected_coords[:,0], selected_coords[:,1] ] = 1
                
                process_component(sub_component, results, min_area)

    return results
    
    
    
def process_component(component, results, min_area, expand=0.5):


    component_img = (component*255).astype(np.uint8)


    contours,_ = cv2.findContours(component_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    if len(contours)==0:
        return


    cnt=max(contours, key=cv2.contourArea)


    area=cv2.contourArea(cnt)


    if area < min_area:
        return


    if len(cnt)<5:
        return


    ellipse=cv2.fitEllipse(cnt)


    (x,y),(MA,ma),angle=ellipse


    if MA < ma:
        MA,ma=ma,MA
        angle+=90

    MA_expand = MA + 2 * expand
    ma_expand = ma + 2 * expand

    ratio = MA_expand/(ma_expand+1e-8)
        
    shape_type = "circle" if ratio < 1.02 else "ellipse"
    
    results.append({
        "center":(float(x),float(y)),
        "major_axis":float(MA_expand),
        "minor_axis":float(ma_expand),
        "axis_ratio":float(ratio),
        "angle":float(angle),
        "area":float(area),
        "type":shape_type})
        
    
    

def draw_ellipses_mask(shape, results, expand=0.5):
    out = np.zeros(shape, dtype=np.float32)

    for r in results:
        center = tuple(map(int, r["center"]))
        axes = (
            int((r["major_axis"] / 2) + expand),
            int((r["minor_axis"] / 2) + expand)
        )
        angle = r["angle"] % 180 
        # OpenCV does not record the direction of the major and minor axes, so 0° and 180° represent the same ellipse. 
        # This normalization is optional, but is useful when analyzing the orientation distribution of ellipses.
        print(r["major_axis"], r["minor_axis"], axes)
        
        cv2.ellipse(out, center, axes, angle, 0, 360, 1, -1)

    return out



def remove_close_peaks(local_max, dist_transform, min_distance=8):

    coords = np.argwhere(local_max)

    if len(coords)==0:
        return local_max

    values = dist_transform[local_max]

    order = np.argsort(values)[::-1]

    keep=[]

    for i in order:

        p = coords[i]

        if len(keep)==0:
            keep.append(i)
            continue

        d = np.linalg.norm(coords[keep]-p, axis=1)

        if np.all(d > min_distance):
            keep.append(i)

    new_local_max = np.zeros_like(local_max)

    for i in keep:
        y,x=coords[i]
        new_local_max[y,x]=True


    return new_local_max


if __name__ == "__main__":
    main()
