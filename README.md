# 🚀CGDF: Constrained Grasp Diffusion Fields

> [IROS 2024](http://iros2024-abudhabi.org/)

[Gaurav Singh](https://vanhalen42.github.io/)* <sup>**1**</sup>, [Sanket Kalwar](https://sanketkalwar.github.io/)* <sup>**1**</sup>, [Md Faizal Karim](https://researchweb.iiit.ac.in/~md.faizal/)<sup>**1**</sup>, [Bipasha Sen](https://bipashasen.github.io/)<sup>**2**</sup>, [Nagamanikandan Govindan](https://nagamanigi.wixsite.com/home)<sup>**1**</sup>, [Srinath Sridhar](https://cs.brown.edu/people/ssrinath/)<sup>**3**</sup>, [K Madhava Krishna](https://scholar.google.com/citations?user=QDuPGHwAAAAJ)<sup>**1**</sup>

*denotes equal contribution, <sup>**1**</sup> International Institute of Information Technology Hyderabad, <sup>**2**</sup> MIT CSAIL, <sup>**3**</sup> Brown University

This is the official implementation of the paper _"Constrained 6-DoF Grasp Generation on Complex Shapes for Improved Dual-Arm Manipulation"_ accepted at **IROS 2024**

## Installation

### Clone the repository

The pretrained checkpoint can be found [here](https://drive.google.com/file/d/1JFQCGPex_36fslssHaKDhsTF2vsy51rZ/view?usp=sharing). Place it in the `demo/data/models/cgdf_v1` directory. Please clone the repository as follows:
```
git clone https://github.com/constrained-grasp-diffusion/constrained-grasp-diffusion.git
```

### CREATING THE ENVIRONMENT

```python
conda create --name cgdf -y python=3.8
conda activate cgdf
bash build_env.sh
```
## Running the Demo

```bash
# Uniform grasp generation
CUDA_VISIBLE_DEVICES=0 python scripts/sample/generate_6d_grasp_poses.py --n_grasps 300 --model cgdf_v1 --input demo/data/meshes/15847850d132460f1fb05d58f51ec4fa.obj  

# Part-constrained grasp generation
CUDA_VISIBLE_DEVICES=0 python scripts/sample/generate_part_guided_6d_grasp_poses.py --n_grasps 300 --model cgdf_v1 --input demo/data/meshes/15847850d132460f1fb05d58f51ec4fa.obj

```
The generated grasps are saved as a mesh containing the object and gripper markers in `output_mesh.obj`.

## 👏 Acknowledgement

This repository is heavily based on [grasp_diffusion](https://github.com/robotgradient/grasp_diffusion) and also borrows code from [Convolutional Occupancy Networks](https://github.com/autonomousvision/convolutional_occupancy_networks). We thank the authors for releasing their code.

## 📜 BibTeX

If you find our work useful, please consider citing us!

```
@article{singh2024constrained,
  title={Constrained 6-DoF Grasp Generation on Complex Shapes for Improved Dual-Arm Manipulation},
  author={Singh, Gaurav and Kalwar, Sanket and Karim, Md Faizal and Sen, Bipasha and Govindan, Nagamanikandan and Sridhar, Srinath and Krishna, K Madhava},
  journal={IROS 2024},
  year={2024}
}
```
