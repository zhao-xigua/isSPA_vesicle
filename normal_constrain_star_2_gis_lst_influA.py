import numpy as np
import argparse
import math

def main():
    parser = argparse.ArgumentParser(description="Screen particles by comparing vesicle normals and particle orientations.")
    parser.add_argument("constrain_star", type=str, help="input 'normal_constrained_rln31.star' file")
    parser.add_argument("recenter_star", type=str, help="input 'BBR_recenter.star' file")
    parser.add_argument("raw_gis_lst", type=str, help="input 'your_raw_gis.lst' file, after removed center and euler duplicates")
    parser.add_argument("matched_gis_lst", type=str, help="output 'normal_matched_raw_gis.lst' file")
    
    args = parser.parse_args()
    
    particle_orients1, header_lines1  = parse_rln31_star_file(args.constrain_star)
    particle_orients2, header_lines2  = parse_rln31_star_file(args.recenter_star)
    
    matched_indices = find_matched_star_index_accord_orients(particle_orients2, particle_orients1, tol = 1e-3)
    
    lst_orients = parse_lst_file(args.raw_gis_lst)
    
    final_lines = []
    for micrograph_name, matched_idxs in matched_indices.items():
        for idx in matched_idxs:
            if idx != -1:  
                full_line = lst_orients[micrograph_name][idx][-1]
                final_lines.append(full_line)
            
    
    with open(args.matched_gis_lst, 'w') as f:
        for line in final_lines:
            f.write(line + '\n')
            

def parse_rln31_star_file(star_path):
    particle_orients = {}
    header_lines = []
    with open(star_path, 'r') as f:
        lines = f.readlines()
        mline=judge_mline0(lines)
        header_lines = lines[:mline]  
        
        for line in lines[mline:]:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split()
                micrograph_path = parts[0]
                micrograph_name = micrograph_path.split('/')[-1].split('.mrc')[0]
                image_num = parts[3].split('@')[0]
                image_sig = image_num + "@"
                x = float(parts[1])
                y = float(parts[2])
                rot = float(parts[9])
                tilt = float(parts[10])
                psi = float(parts[11])
                
                if micrograph_name not in particle_orients:
                    particle_orients[micrograph_name] = []
                particle_orients[micrograph_name].append((image_sig, x, y, rot, tilt, psi))
    return particle_orients, header_lines
    
    
def parse_lst_file(raw_gis_lst):
    raw_gis_particles = {}
    with open(raw_gis_lst, 'r') as f:
        for line in f:
            parts = line.strip().split()
            micrograph_path2 = parts[1]
            center_x = float(parts[6].split('=')[1].split(',')[0])
            center_y = float(parts[6].split('=')[1].split(',')[1])
            euler1 = float(parts[5].split('=')[1].split(',')[0])
            euler2 = float(parts[5].split('=')[1].split(',')[1])
            euler3 = float(parts[5].split('=')[1].split(',')[2])
            rot = math.fmod(euler2-90,360)
            tilt = euler1
            psi = math.fmod(euler3+90,360)
            micrograph_name2 = micrograph_path2.split('/')[-1].split('.mrc')[0]
            if micrograph_name2 not in raw_gis_particles:
                raw_gis_particles[micrograph_name2] = []
            raw_gis_particles[micrograph_name2].append((center_x, center_y, rot, tilt, psi, line.strip()))
    
    return raw_gis_particles
    
    
def find_matched_star_index_accord_orients(raw_gis_star_dicts, matched_extract_star_dicts, tol = 1e-3):
    """
    Compare raw_gis_star_dicts and matched_extract_star_dicts, retrieve each entry in matched_extract_star_dicts.
    Matching index in raw_gis_star_dicts
    """
    matched_indices = {}

    for micrograph_name2, entry2_list in matched_extract_star_dicts.items():
        for sig2, x2, y2, rot2, tilt2, psi2 in entry2_list:
            if micrograph_name2 in raw_gis_star_dicts:
                for idx, entry1 in enumerate(raw_gis_star_dicts[micrograph_name2]):
                    sig1, x1, y1, rot1, tilt1, psi1 = entry1
                    if ( abs(x1-x2)<tol and abs(y1-y2)<tol and abs(rot1-rot2)<tol and abs(tilt1-tilt2)<tol and abs(psi1-psi2)<tol ):
                        if micrograph_name2 not in matched_indices:
                            matched_indices[micrograph_name2] = []
                        matched_indices[micrograph_name2].append(idx)
                        break  
                    else:
                        if micrograph_name2 not in matched_indices:
                            matched_indices[micrograph_name2] = []
                        matched_indices[micrograph_name2].append(-1)

    return matched_indices
    

def judge_mline0(inline):
    trys=60
    intarget=-1
    n=1
    limit=20
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
                    
                    
if __name__ == "__main__":
    main()
