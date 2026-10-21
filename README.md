# isSPA-vesicle

Welcome to use isSPA-vesicle! A tool for vesicle geometry-guided post hoc filtering of particles in .star files, especially for isSPA-detected particles. 

## 1. Functions and Workflow

very simple:

![very_simple](images/very_simple.jpg)

full workflow:

![full_workflow](images/full_workflow.jpg)

associated scripts:

![associated_scripts](images/associated_scripts.jpg)

## 2. Installation

We recommend creating an isolated virtual environment to avoid conflicts with existing Python environments on your machine. Below are `conda` environment setup instructions. 

```shell
conda create -n isSPA_vesicle python=3.10 -y
conda activate isSPA_vesicle 
pip install -r requirements.txt
```

Run any `isSPA_vesicle_xxx.py` script with `-h` to verify the environment configuration. If all scripts can execute, the environment has been installed successfully.

## 3. Quick Start

### 3.1 Image Pre-processing (required)

```shell
mkdir image
proc2d input.mrc output.mrc apix=xxx lp=20 hp=800 meanshrink=10 
cp output_*.mrc image/
```

The pre-processing command is based on EMAN. This operation is also available in other software including EMAN2 and RELION. After binning by a factor of 10, the images are typically around 400 × 400 pixels. These images can be used directly as GUI input, or preprocessed to 512 × 512 pixels before input. Both options are suitable.

### 3.2 GUI manual labeling (optional)

```shell
python isSPA_vesicle_GUI.py --bin 10 &

--bin {5,10}   ### 5 or 10 according to the micrographs you binned
```

After labeling one micrograph and saving the associated files, close the GUI. Rerun the same command to label the next image. After labeling, the image and mask are automatically padded or cropped to 512 × 512 pixels.

**GUI Controls**
Mouse wheel: Zoom in/out of the micrograph

Left mouse button: Click to confirm vesicle position

Axes(a,b) and Angle(°): Adjust the shape of the fitted ellipse, then press Enter

Mask Name: Enter the rootname without file extension

![GUI_image](images/GUI_cut_from_zmj_destop.png)

![GUI_zoom](images/GUI_zoom.png)

### 3.3 AI Model Train (optional)

```shell
python isSPA_vesicle_Unet.py 
--mrc_dir image            ## preprocessed images dir (size:512×512)
--mask_dir mask            ## manual labeled masks dir (size:512×512)
--output_dir results       ## for saving predictions of validated images
--epochs 500               ## suitable for most datasets
--batch_size 4             ## tune according to vesicles/micrograph
--gpu 0                    ## use a single gpu_id
```

`epochs` generally no adjustment required.

`batch_size` For micrographs with many vesicles, use a smaller `batch_size` (e.g., 2). For micrographs with only 1–3 target vesicles, use a larger `batch_size` (e.g., 8–12). Values above 12 are not recommended due to potential out-of-memory errors.

### 3.4 Vesicle Predict (required)

```shell
python isSPA_vesicle_Unet.py
--mrc_dir image               ## unused in prediction,but can be retained 
--mask_dir mask               ## unused in prediction
--output_dir predict_results  ## for saving predicted masks
--epochs 500                  ## unused in prediction
--batch_size 4                ## unused in prediction
--predict                     ## if predicts, select it
--predict_dir predict_image   ## for providing images to be predicted
--model_path unet_model.pth   ## the model you believe best
--gpu 0                       ## use a single gpu_id
```

During model prediction, only `output_dir`, `predict`, `predict_dir` and `model_path` are required. The remaining parameters are not used during prediction.

### 3.5 Fit Ellipses (required)

```shell
python isSPA_vesicle_fit_ellipsoid_auto.py 
--input predict_mask.mrc                  ## single mrc file
--output_txt predict_mask_fit_ellip.txt   ## fit vesicles, get para text
--output_mrc predict_mask_fit_ellip.mrc   ## fit vesicles, get fit image
--min_area 400                            ## filter noise areas(pixels^2)
--peak_size 18                            ## tune accord vesicle size
--min_distance 8                          ## tune accord vesicle size
```

`min_area` specifies the minimum connected-component area (in pixels²) retained during noise filering in a 512 × 512 image. Set this threshold according to the target vesicle size. Use a lower value for smaller vesicles to avoid excluding them.

For larger target vesicles, `peak_size` and `min_distance` may need to be increased. The default settings have been tested on the AcrB dataset (EMPIAR-10426).

### 3.6 Scale Info (required)

```shell
python isSPA_vesicle_scale_info.py 
input_file                     ## input predict_mask_fit_ellip.txt
output_file                    ## output predict_mask_fit_ellip_scale.txt
--ms10_mic_size 409 409        ## bin10 image size
--micrograph_size 4096 4096    ## original micrograph size
```

This step scales the vesicle parameter file generated from 512 × 512 images to match the dimensions of the original micrographs.

### 3.7 Distance_from_mem Measure (required)

```shell
python isSPA_vesicle_distance_measure_tool.py
--filename output_cylinder.mrc         ## output filename                  
--shape 200 200 200                    ## map boxsize
--voxel_size 1                         ## map voxel size
--radius 30                            ## cylinder radius
--z_range 30 70                        ## cylinder lower and upper bounds
```

The lipid bilayer membrane is usually ~40 Å thick (Nagle, J. F., & Tristram-Nagle, S. 2000); the span of `z_range` can be calculated from the `voxel_size`. The units of these parameters `shape`,  `radius`, `z_range` and `distance_from_mem` are pixels.

Set `shape` and `voxel_size` to match those of the 3D map used for particle picking.

The `radius` usually requires little adjustment and should reflect the spatial dimensions of the target protein’s transmembrane region.

Carefully adjust `z_range` until the generated cylinder density fully matches the transmembrane region of the membrane protein. The resulting `distance_from_mem` value can then be used in `isSPA_vesicle_constrain_optimise.py`.

### 3.8 Filtering Particles (required)

```shell
python isSPA_vesicle_constrain_optimise.py
vesicle_info vesicle_scale.txt        ## vesicle info from scale.py
star_path particles.star              ## allow RELION 3.0 and 3.1 star 
output_star filtered_particles.star   ## same format with the input star
--radius_scale 1.2                    ## usually fixed
--distance_thresh 15                  ## unit (pixels)
--angle_thresh 90                     ## unit (°)
--distance_from_mem -25               ## inward protein (negative value)
--euler_ori 1                         ## choose {1,-1}
--protein_orient -1                   ## choose {1,-1}
--prefix stack                        ## common prefix of image names
--suffix ""                           ## common suffix of image names
```

`radius_scale` generally no adjustment required.

`distance_thresh` This threshold can be set slightly more lenient than the typical cutoff used to identify duplicate particles.

`angle_thresh` Relates to the tilt of membrane proteins within the membrane. See Results and Methods in the paper for settings.

`distance_from_mem` This value is negative for membrane proteins anchored inward toward the vesicle lumen, and positive for proteins anchored outward from the vesicle. The value can be calculated using `isSPA_vesicle_distance_measure_tool.py`. It defines the distance (in pixels) between the box center of the 3D particle‑picking template and the membrane surface on the protein-facing side. 

`euler_ori` {1,-1} 1: initial template ECD region faces the viewer; -1: transmembrane region faces the viewer (in Chimera)

`protein_orient` {1,-1} 1: protein oriented toward vesicle exterior (outward-facing); -1: protein oriented toward vesicle interior (inward-facing)

`prefix`and`suffix` These are optional parameters and are automatically estimated internally by the software. Automatic calculation may fail for filenames with complex naming conventions. User may manually specify the prefix and suffix for filename matching. Example: 

**particles.star (file header is not displayed)**

```
MotionCorr/job008/micrographs/LP1/stack_1st_0001.mrc	1118.0	3219.0	000001@MotionCorr/job008/micrographs/LP1/stack_1st_0001.mrc	15624.57	16378.07	82.910408	300.000000	0.01
	0.100000	44883.3034111	5.000000	0.140000	1	-46.143101	101.967598	111.0	0.000000	0.000000	1	0.600000	1.800000e+05	1.000000	1
MotionCorr/job008/micrographs/LP1/stack_1st_0001.mrc	2690.0	1587.0	000001@MotionCorr/job008/micrographs/LP1/stack_1st_0001.mrc	15624.57	16378.07	82.910408	300.000000	0.01
	0.100000	44883.3034111	5.000000	0.140000	1	-43.0662	101.967598	255.0	0.000000	0.000000	1	0.600000	1.800000e+05	1.000000	1
```

**vesicle_scale.txt**

```
stack_1st_0001.mrc	image_size=3710,3838	center=1884.10,21.70	radius=114.00,98.20	angle=141.19	axis_ratio=1.16	type=ellipse
stack_1st_0001.mrc	image_size=3710,3838	center=3247.40,121.90	radius=194.90,114.40	angle=169.94	axis_ratio=1.70	type=ellipse
```

The first column of the two files shows `stack_xxx_xxxx.mrc` as the shared filename segment. So, `--prefix stack`  `--suffix ""` are suitable.

Vesicle geometry-based particle filtering is now complete! The output STAR file retains the same format as the input STAR file and is ready for downstream processing.

## 4. Example Data and trained AI model

Download the test datasets from Zenodo (DOI: [10.5281/zenodo.23470674](https://doi.org/10.5281/zenodo.23470674)) and the trained AI model from Zenodo (DOI: [10.5281/zenodo.23444711](https://doi.org/10.5281/zenodo.23444711)).