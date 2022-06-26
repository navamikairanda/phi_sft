# Φ-SfT: Shape-from-Template with a Physics-Based Deformation Model
### [Project Page](https://4dqv.mpi-inf.mpg.de/phi-SfT/) | [Video](https://youtu.be/2jxDq8qyfg8) | [Paper](https://arxiv.org/pdf/2203.11938.pdf) | [Data](https://drive.google.com/drive/folders/1gpzp5k64S6TnDbl8ZW8lgSmDE_nzHdh9)

[Navami Kairanda](https://people.mpi-inf.mpg.de/~nkairand/),
[Edith Tretschk](https://people.mpi-inf.mpg.de/~tretschk/),
[Mohamed Elgharib](https://people.mpi-inf.mpg.de/~elgharib/),
[Christian Theobalt](https://people.mpi-inf.mpg.de/~theobalt/),
[Vladislav Golyanik](https://people.mpi-inf.mpg.de/~golyanik/) <br>
Max Planck Institute for Informatics <br>
in CVPR 2022

This is the official implementation of the paper "Φ-SfT: Shape-from-Template with a Physics-Based Deformation Model".

[![Watch the video](https://img.youtube.com/vi/2jxDq8qyfg8/hqdefault.jpg)](https://youtu.be/2jxDq8qyfg8)

## What is Φ-SfT?
Φ-Sft is an analysis-by-synthesis method to reconstruct a temporally coherent sequence of 3D shapes from a monocular RGB video, given a single initial 3D template in advance. The method models deformations as a response of the elastic surface to the external and internal forces acting on it. Given the initial 3D template, tt uses a physics simulator to generate deformed future states, which are treated as the reconstructed surfaces. The set of physical parameters Φ that describe the challenging surface deformations are optimised by minimising a dense per-pixel photometric energy. We backpropagate through the differentiable rendering and differentiable physics simulator to obtain the optimal set of physics parameters Φ. 

## Installation

Clone this respository to `${code_root}`. The following sets up a conda environment with all Φ-SfT dependencies

```
conda env create -f ${code_root}/phi_sft/environment.yml
conda activate phi_sft
```

Φ-SfT uses physics simulator as a deformation model prior. Here, we package and re-distribute such a differentiable physics simulator.
You can first build the [arcsim](http://graphics.berkeley.edu/resources/ARCSim/) dependencies (ALGLIB, JsonCpp, and TAUCS) with
```
cd ${code_root}/arcsim/dependencies; make
```
This assumes that you have the following libraries installed. On Linux, you should be able to get all of them through your distribution's package manager.
* BLAS
* Boost
* freeglut
* gfortran
* LAPACK
* libpng

After arcsim setup, you can compile the [differentiable physics simulator](https://github.com/williamljb/DifferentiableCloth) by going back to the parent directory, updating paths in `setup.py` and running 'make'. 
```
cd ${code_root}; make
```
One can verify that the simulator installation is successful using
```
import torch
import arcsim
```

## Running code

### Reconstructing 3D shapes from monocular sequences
Download and extract Φ-SfT [dataset](https://drive.google.com/drive/folders/1gpzp5k64S6TnDbl8ZW8lgSmDE_nzHdh9) to `${data_root}`. For reconstructing real sequence S3, update paths (and other parameters) in experiment configuration file `${code_root}/phi_sft/config/expt_real_s3.ini`, data sequence configuration file `${data_root}/real/S3/preprocess.ini` and simulator configuration file `${data_root}/real/S3/sim_conf.json`

```
sequence_type=real
sequence_name=s3
cd ${code_root}/phi_sft
conda activate phi_sft
python -u reconstruct_surfaces_from_sequence.py ${data_root}/${sequence_type} ${code_root}/phi_sft/config/expt_${sequence_type}_${sequence_name}.ini
```
The reconstruction process requires several hundred iterations in most cases, which takes 16−24 hours on an Nvidia RTX 8000 GPU for roughly 50 frames of 1920x1080 image sequences and template mesh ~300 vertices. During recontruction at every `i_save` iterations, the code will save checkpoints and evaluate quantitatively against the groundtruth and store the result in log.txt. 

To resume training from last checkpoint (e.g. iteration 100), set `reload=False` and `i_reload=100` in experiment configuration file `${code_root}/phi_sft/config/expt_real_s3.ini` and run `reconstruct_surfaces_from_sequence.py ` script above

### Preparing real sequences

If you would like to try Φ-SfT on your own dataset, create a folder `${sequence_name}` in `${data_root}/real` with the following structure
* `rgbs` with monocular RGB images of a deforming surface
* `point_clouds` with point cloud for 3D template of the surface corresponding to the first frame
* `masks` segmentation masks
* `preprocess.ini`, configuration parameters for the sequence
* `camera.json`, camera intrinsics assuming the camera coordinate system of [Kinect SDK](https://docs.microsoft.com/en-us/azure/kinect-dk/coordinate-systems) 
* `sim_conf.json`, simulation configuration, can be kept unchanged

The blurred version of ground-truth segmentation masks are used for silhouette loss in Φ-SfT, and can be generated with  
```
python preprocess_real_sequence.py  ${data_root}/real ${sequence_name} blur_masked_images 
```

Given Kinect RGB image and point cloud for template, generate 3D template with poisson surface reconstruction. The corresponding texture map for the template is obtained by projecting the vertices of the template mesh onto the image space of the ﬁrst image with known camera intrinsics.
```
python preprocess_real_sequence.py  ${data_root}/real ${sequence_name} generate_template_surface
```

The generated template can be cleaned with a mesh editing tool such as Blender and saved as `${data_root}/real/${sequence_name}/templates/template_mesh_final.obj`. The following script prepares the template in the format expected by the physics simulator used in Φ-SfT. It determines the initial rigid pose of the template relative to a ﬂat sheet on XY plane (as requried in arcsim) using iterative closest point (ICP). Pose estimation parameters in `${data_root}/real/${sequence_name}/preprocess.ini` can be modified to get accurate pose.
```
python preprocess_real_sequence.py  ${data_root}/real ${sequence_name} clean_template_surface
```

Now, your data sequence is ready for running Φ-SfT! Create an experiment configuration file in `{code_root}/phi_sft/config` and follow the instructions in the previous section. If the generated template is not positively oriented, you may get the follwing error `Error: TAUCS failed with return value -1` while running `reconstruct_surfaces_from_sequence.py`. This can be addressed by setting invert_faces_orientation=True,
 in `preprocess.ini`and running the last step in template processing (clean_template_surface). 

### Generating synthetic sequences
If you would like to generate synthetic datasets with physics simulator, create a folder `${sequence_name}` in `${data_root}/synthetic` with following structure
* `preprocess.ini`, configuration parameters for the sequence
* `camera.json`, camera intrinsics assuming the camera coordinate system of PyTorch3D 
* `sim_conf.json`, simulation configuration, please modify the forces (gravity, wind) and material here to generate interesting surface deformations

The following commands generate challenging synthetic surface and image sequence

```
python generate_synthetic_sequence.py ${data_root}/synthetic ${sequence_name} generate_surfaces
python generate_synthetic_sequence.py ${data_root}/synthetic ${sequence_name} render_surfaces
```

Now, your data sequence is ready for running Φ-SfT, create an experiment configuration file in `{code_root}/phi_sft/config` and follow the instructions in the first section.
 
### Evaluation

Run the following to evaluate the reconstructed surfaces of Φ-SfT againt ground-truth. This computes the chamfer distance for real data sequences and angular as well as 3D error for synthetic sequences. Additionally, it provides mesh and depth visualisations of reconstructed surfaces as in the paper.

```
sequence_type=real
sequence_name=s3
converged_iteration=300

python evaluate_reconstructed_surfaces.py ${data_root}/${sequence_type} config/expt_${sequence_type}_${sequence_name}.ini $converged_iteration
```

## Citation

If you use this code for your research, please cite:
```
@inproceedings{kair2022sft,
	title={$\phi$-SfT: Shape-from-Template with a Physics-Based Deformation Model},
	author={Navami Kairanda and Edith Tretschk and Mohamed Elgharib and Christian Theobalt and Vladislav Golyanik},
	booktitle={Computer Vision and Pattern Recognition (CVPR)},
	year={2022}
}
@inproceedings{liang2019differentiable,
	title={Differentiable Cloth Simulation for Inverse Problems},
	author={Junbang Liang and Ming C. Lin and Vladlen Koltun},
	booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
	year={2019}
}

```


## License
This software is provided freely for non-commercial use. The code builds on the differentiable version (https://github.com/williamljb/DifferentiableCloth) of ARCSim cloth simulator (http://graphics.berkeley.edu/resources/ARCSim/). We thank both of them for releasing their code.

We release this code under MIT license. You can find all licenses in the file LICENSE.