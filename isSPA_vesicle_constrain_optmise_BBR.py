import numpy as np
import argparse
from scipy.spatial import cKDTree
import math
from tqdm import tqdm
import warnings
import os
from generate_sampling_points_and_normal import auto_sampling_density, generate_sphere_points, map_points_to_ellipsoid


def main():
    parser = argparse.ArgumentParser(description="Filter particles by comparing vesicle normals and particle orientations, \nAccepted relion31 and relion30.star")
    parser.add_argument("vesicles_info", type=str, help="input 'vesicles_info' file, including vesicless' center and radiu")
    parser.add_argument("star_path", type=str, help="input 'particles.star' file")
    parser.add_argument("output_star", type=str, help="output 'filtered_particles.star' file")
    parser.add_argument('--radius_scale', type=float, default=1.2, help='vesicle radius scaling factor. default=1.2')
    parser.add_argument('--distance_thresh', type=float, default=15.0, help='center range')
    parser.add_argument('--angle_thresh', type=float, default=15.0, help='angle range')
    parser.add_argument('--euler_ori',type=int,choices=[1, -1], help='1: initial template head faces the viewer; -1: transmembrane region faces the viewer (in Chimera)')
    parser.add_argument('--protein_orient',type=int, choices=[1, -1], default=1, help='1: protein oriented toward vesicle exterior (outward-facing); \n-1: protein oriented toward vesicle interior (inward-facing)')
    parser.add_argument('--prefix', type=str, default=None, help='manual filename prefix for matching micrographs')
    parser.add_argument('--suffix', type=str, default=None, help='manual filename suffix for matching micrographs')
    
    args = parser.parse_args()
    
    particle_orients, header_lines  = parse_star_file(args.star_path)
    all_names = list(particle_orients.keys())
    
    # Use user input if provided
    if args.prefix is not None or args.suffix is not None:

        if args.prefix is None or args.suffix is None:
            raise ValueError(
                "Both --prefix and --suffix must be provided together"
            )

        prefix = args.prefix
        suffix = args.suffix

        print("[INFO] Using manual prefix/suffix:")
        print("       prefix =", prefix)
        print("       suffix =", suffix)
        
    else:

        prefix, suffix = find_common_prefix_suffix(all_names)

        print("[INFO] Auto detected prefix/suffix:")
        print("       prefix =", prefix)
        print("       suffix =", suffix)

    vesicles = parse_vesicles_lst_file(args.vesicles_info, prefix, suffix)
    
    # Validate filenames when no common suffix is available
    if suffix == "":
        validate_rootname_lengths(vesicles, all_names)
    
    retained_lines, filtered_vesicles = filter_vesicles(particle_orients, vesicles, args)
    
    
    euler_ori_vec = np.array([0, 0, args.euler_ori])
    min_dist = 6
    all_final_lines = []
    
    for micrograph_name, ves_list in tqdm(filtered_vesicles.items(), total=len(filtered_vesicles), desc="Matching:", unit="Vesicles"):
        
        particles = particle_orients.get(micrograph_name, [])
        
        for ves in ves_list:
            center_x, center_y, radius_long, radius_short, angle, raw = ves
            
            axes = (radius_long, radius_short, radius_short)
            center = (center_x, center_y, 0)
            
            anticipating_nside = auto_sampling_density(axes, min_dist)
    
            sphere_pts, normals = generate_sphere_points(anticipating_nside)
        
            q_global, q_normals = map_points_to_ellipsoid(sphere_pts, axes=axes, ellipsoid_center=center, angle_deg=angle, protein_orient=args.protein_orient)

            matched_indices = match_particles_with_kdtree(particles, ves, q_global, q_normals, args.distance_thresh, args.angle_thresh, euler_ori_vec)
            #print(matched_indices)
            
            for idx in matched_indices:
                all_final_lines.append(particles[idx][-1])
            
            
    # write to .star file
    with open(args.output_star, 'w') as f:
        f.writelines(header_lines)
        for line in all_final_lines:
            f.write(line + '\n')
    

def validate_rootname_lengths(vesicles, star_names, sample_size=10):
    """
    Compare rootname lengths using up to sample_size names
    from the vesicles dictionary and STAR.
    """
    vesicle_rootnames = list(vesicles.keys())[:sample_size]
    star_rootnames = list(star_names)[:sample_size]

    if not vesicle_rootnames or not star_rootnames:
        raise ValueError(
            "Cannot validate rootname lengths: "
            "no filenames found in vesicles or STAR."
        )

    vesicle_lengths = {len(name) for name in vesicle_rootnames}
    star_lengths = {len(name) for name in star_rootnames}

    if vesicle_lengths != star_lengths:
        raise ValueError(
            "\n[ERROR] Filename format mismatch detected.\n"
            f"Rootname lengths in vesicles: {sorted(vesicle_lengths)}\n"
            f"Rootname lengths in STAR: {sorted(star_lengths)}\n"
            "Please ensure that filenames in the first column of "
            "vesicles_info use the same naming format as filenames "
            "in the STAR file."
        )

    print(
        "[INFO] Filename format validation passed. "
        f"Rootname length(s): {sorted(star_lengths)}"
    )

    
def parse_star_file(star_path):
    particle_orients = {}
    header_lines = []
    column_indices = {}
    
    with open(star_path, 'r') as f:
        lines = f.readlines()
        mline=judge_mline0(lines)
        header_lines = lines[:mline]  
        
        for line in header_lines:
            line = line.strip()
            if line.startswith('_rln'):
                parts = line.split()
                key = parts[0]
                if len(parts) > 1 and parts[1].startswith('#'):
                    idx = int(parts[1][1:]) - 1
                    column_indices[key] = idx
                    
        required_keys = ['_rlnMicrographName', '_rlnCoordinateX', '_rlnCoordinateY','_rlnAngleRot', '_rlnAngleTilt', '_rlnAnglePsi']
        
        for key in required_keys:
            if key not in column_indices:
                raise ValueError(f"Cannot find required column: {key} in {star_path}")
        
        for line in lines[mline:]:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                micrograph_path = parts[column_indices['_rlnMicrographName']]
                x = float(parts[column_indices['_rlnCoordinateX']])
                y = float(parts[column_indices['_rlnCoordinateY']])
                micrograph_name = micrograph_path.split('/')[-1].split('.mrc')[0]
                rot = float(parts[column_indices['_rlnAngleRot']])
                tilt = float(parts[column_indices['_rlnAngleTilt']])
                psi = float(parts[column_indices['_rlnAnglePsi']])
                if micrograph_name not in particle_orients:
                    particle_orients[micrograph_name] = []
                particle_orients[micrograph_name].append((x, y, rot, tilt, psi, line.strip()))
        
        all_names = list(particle_orients.keys())
        common_prefix, common_suffix = find_common_prefix_suffix(all_names)

        print(f"[INFO] Common prefix: {common_prefix}")
        print(f"[INFO] Common suffix: {common_suffix}")
        
    return particle_orients, header_lines
    
    
def parse_vesicles_lst_file(vesicles_info, prefix, suffix):
    vesicles = {}
    warned = False
    failed_match = []
    
    with open(vesicles_info, 'r') as f:
        for line in f:
            parts = line.strip().split()
            micrograph_path2 = parts[0]
            
            center_x = float(parts[2].split('=')[1].split(',')[0])
            center_y = float(parts[2].split('=')[1].split(',')[1])
            radius_long = float(parts[3].split('=')[1].split(',')[0])
            radius_short = float(parts[3].split('=')[1].split(',')[1])
            angle = float(parts[4].split('=')[1])
            
            micrograph_name2 = extract_name_by_anchor(micrograph_path2, prefix, suffix)
            
            if micrograph_name2 is None:
                failed_match.append(micrograph_path2)
                continue
                
            
            if len(failed_match)>0:

                print("\n[ERROR] Cannot match following vesicles:")

                for x in failed_match[:10]:
                    print("   ",x)

                raise RuntimeError(
                    "Filename matching failed. "
                    "Please provide --prefix and --suffix manually."
                )
            
            if micrograph_name2 not in vesicles:
                vesicles[micrograph_name2] = []
            vesicles[micrograph_name2].append((center_x, center_y, radius_long, radius_short, angle, line.strip()))
    
    return vesicles
            
    

def is_inside_ellipsoid(x, y, cx, cy, r_long_scaled, r_short_scaled, angle):
    """
    Check whether point (x, y) lies inside a rotated ellipse
    angle: same convention as cv2.ellipse (degrees)
    """

    # 1. translation to ellipse center
    dx = x - cx
    dy = y - cy

    # 2. angles to rad, and invert
    theta = math.radians(-angle)

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # 3. rotation
    x_rot = dx * cos_t - dy * sin_t
    y_rot = dx * sin_t + dy * cos_t

    # 4. ellipse equation validation
    value = (x_rot**2) / (r_long_scaled**2) + (y_rot**2) / (r_short_scaled**2)

    return value <= 1.0
    

def filter_vesicles(particle_orients, vesicles, args):
    retained_lines = []
    filtered_vesicles = {}
    
    for image_name, vesicle_list in vesicles.items():
        
        if image_name in particle_orients:
            
            for cx, cy, r_long, r_short, angle, original_line in vesicle_list:
                r_long_scaled = r_long * args.radius_scale
                r_short_scaled = r_short * args.radius_scale
                
                for px, py, rot, tilt, psi, _ in particle_orients[image_name]:
                    if is_inside_ellipsoid(px, py, cx, cy, r_long_scaled, r_short_scaled, angle):
                        retained_lines.append(original_line)
                        filtered_vesicles.setdefault(image_name, []).append((cx, cy, r_long, r_short, angle, original_line))
                        break            # Once a protein is found in a vesicle, keep this vesicle and move to the next one
    
    return retained_lines, filtered_vesicles


def match_particles_with_kdtree(particles, ves, q_global, q_normals, distance_thresh, angle_thresh, euler_ori_vec):
    matched = []
    
    # --- 1. Build KDTree from ellipsoid sampling points ---
    
    projected_coords = q_global[:, :2]     
    projected_normals = q_normals
    
    if len(projected_coords) == 0:
        return []
    
    tree = cKDTree(projected_coords)

    # --- 2. Iterate over particles ---
    for idx, (x, y, rot, tilt, psi, _) in enumerate(particles):
    
        # RELION euler → vectors
        star_rot_rad = math.radians(rot)
        star_tilt_rad = math.radians(tilt)
        star_psi_rad = math.radians(psi)
        
        rotmat = [[1,2,3],[4,5,6],[7,8,9]]
        #print(rotmat)
        c1 = math.cos(star_rot_rad)
        c2 = math.cos(star_tilt_rad)
        c3 = math.cos(star_psi_rad)
        s1 = math.sin(star_rot_rad)
        s2 = math.sin(star_tilt_rad)
        s3 = math.sin(star_psi_rad)
        rotmat[0][0] = c1*c2*c3 - s1*s3
        rotmat[0][1] = -1*c3*s1 - c1*c2*s3
        rotmat[0][2] = c1*s2
        rotmat[1][0] = c1*s3 + c2*c3*s1
        rotmat[1][1] = c1*c3 - c2*s1*s3
        rotmat[1][2] = s1*s2
        rotmat[2][0] = -1*c3*s2
        rotmat[2][1] = s2*s3
        rotmat[2][2] = c2
        rotmatT=np.transpose(rotmat)
        
        particle_normal = np.dot(rotmatT,euler_ori_vec)
        
        
        indices = tree.query_ball_point([x, y], distance_thresh)   # Returned indices correspond to tree array indices
        if isinstance(indices, int): 
            indices = [indices]

        for tree_idx in indices:
            
            heal_normal = projected_normals[tree_idx]
            dot = np.dot(particle_normal, heal_normal)
            angle_rad = np.arccos(np.clip(dot, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)
            
            if angle_deg < angle_thresh:
                matched.append(idx)
                break  
    
            
    return matched
   

def judge_mline0(inline):
    trys=60
    intarget=-1
    n=1
    limit=15
    for i in range (0,trys):
        if(inline[i].split()):
            if(inline[i].split()[0][0]!="_"):
                if(intarget==1):
                    return i
                    break
                else:
                    continue
            if(inline[i].split()[0][0]=="_"):
                if (n < limit):
                    intarget=-1
                    n+=1
                if (n >= limit):
                    intarget=1
    

def find_common_prefix_suffix(names):
    if not names:
        return "", ""
        
    root_names = [os.path.splitext(n)[0] for n in names]
    
    prefix = root_names[0]
    for name in root_names[1:]:
        i = 0
        while i < len(prefix) and i < len(name) and prefix[i] == name[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            break
            
    suffix = root_names[0]
    for name in root_names[1:]:
        i = 0
        while i < len(suffix) and i < len(name) and suffix[-(i+1)] == name[-(i+1)]:
            i += 1
        suffix = suffix[-i:]
        if not suffix:
            break
                
    return prefix, suffix
    
    
def extract_name_by_anchor(path, prefix, suffix):
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    
    start = name.find(prefix)
    end = name.rfind(suffix)
    
    if start == -1 or end == -1:
        return None  
    
    end += len(suffix)
    
    return name[start:end]


if __name__ == "__main__":
    main()
