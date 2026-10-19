

# 1. Installation

1. Download "isSPA_vesicle" package
2.   ```cd isSPA_vesicle/```
3.  Create virtural environment :
```
conda creat -n isSPA_vesicle python=3.10 -y
```
4. Activate :
```
conda activate isSPA_vesicle 
```
5. Install dependencies :
```
pip install -r requirements.txt
```
6.  Run any ```isSPA_vesicle_xxx.py``` script to verify the environment configuration. For example :
```
python isSPA_vesicle_GUI.py --bin 10 
```
# 2. Running scripts

1. Image preparation (in EMAN) :
```
mkdir image/ mask/ results/
```
```
proc3d xxx_dir/input.mrc output.mrc lp=20 hp=800 meanshrink=10 apix=xxx
```
```
cp output.mrc image/
```
Or you can execute these commands one by one.

2. Manual labeling : 
```
python isSPA_vesicle_GUI.py --bin 10 &
```
Scroll the mouse wheel to zoom the image. Left-click to place the ellipse; middle-click to show/hide the elliptical ring.
After manual labeling, save the file as ``` _mask``` with the same root name as the corresponding image in ```mask/```  directory.

3.  Training model :
```
python isSPA_vesicle_Unet_main.py --mrc_dir image/ --mask_dir mask/ --output_dir results/ --epochs 500 --batch_size 8 --gpu 0
```
4. Predict across datasets :
```
python isSPA_vesicle_Unet_main.py --mrc_dir image/ --mask_dir mask/ --output_dir predict_results/ --epochs 500 --batch_size 8 --gpu 0 --predict --predict_dir predict_image/ --model_path unet_model_xxx.pth 
```
5. Get vesicle information, for example :
```
python isSPA_vesicle_scale_info.py predict_results/mask_fit.txt predict_results/mask_fit_scale.txt --ms10_mic_size 409,409 --micrograph_size 4096,4096
```
```
cat predict_results/mask_fit_scale.txt > vesicle_info.txt
```

6. Normal constrain with isSPA/ conventional STAR file :
```
python isSPA_vesicle_constrain.py vesicle_info.txt isSPA_particles_orients.star isSPA_particles_orients_normal.star --distancce_thres 20 --angle_thresh 30 --distance_from_mem xxx --euler_ori 1 --protein_orient 1
```
--distance_from_mem
The distance denotes the offset from the box center to the outer surface of membrane of the lipid bilayer.

7.  "distance_from_mem" measuring tool : 
For example :
```
python isSPA_vesicle_distance_measure_tool.py --filename cylinder.mrc --shape 334 334 334 --voxel_size xxx --radius 20 --z_range 57 77 
```

# Continuing update ...
