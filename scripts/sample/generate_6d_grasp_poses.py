import copy
import configargparse

import scipy.spatial.transform
import numpy as np
from se3dif.models.loader import load_model
from se3dif.samplers import Grasp_AnnealedLD
from se3dif.utils import to_numpy, to_torch
import torch.nn.functional as F

import torch
import os
import trimesh

def parse_args():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--n_grasps', type=str, default='200')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--model', type=str, default='cgdf_v1')
    p.add_argument('--input', type=str, required=True)

    opt = p.parse_args()
    return opt


def get_approximated_grasp_diffusion_field(p, args, device='cpu'):
    model_params = args.model
    batch = int(args.n_grasps)
    ## Load model
    model_args = {
        'device': device,
        'pretrained_model': model_params
    }
    model = load_model(model_args)

    context = to_torch(p[None,...], device)
    model.set_latent(context, batch=batch)

    ########### 2. SET SAMPLING METHOD #############
    generator = Grasp_AnnealedLD(model, batch=batch, T=70, T_fit=50, k_steps=2, device=device)

    return generator, model

def mean_center_and_normalize(mesh):
    mean = mesh.sample(1000).mean(0)
    mesh.vertices -= mean
    scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
    mesh.apply_scale(1/(2*scale))
    return mesh


def sample_pointcloud(input_path=None):
    mesh = trimesh.load(input_path)
    mesh = mean_center_and_normalize(mesh)
    # mesh.export('mesh.obj')

    # sample point cloud
    P = mesh.sample(1000)
    
    # apply random rotation
    sampled_rot = scipy.spatial.transform.Rotation.random()
    rot = sampled_rot.as_matrix()

    P = np.einsum('mn,bn->bm', rot, P)
    P *= 8.
    P_mean = np.mean(P, 0)
    P += -P_mean

    H = np.eye(4)
    H[:3,:3] = rot
    mesh.apply_transform(H)
    mesh.apply_scale(8.)
    H = np.eye(4)
    H[:3,-1] = -P_mean
    mesh.apply_transform(H)

    return P, mesh


if __name__ == '__main__':

    args = parse_args()

    n_grasps = int(args.n_grasps)
    device = args.device
    input_path = args.input

    ## Set Model and Sample Generator ##
    P, mesh = sample_pointcloud(input_path)
    generator, model = get_approximated_grasp_diffusion_field(P, args, device)
    # print(generator)
    # print(model)

    # Running the model
    H_, t = generator.sample()

    H = H_.reshape(-1,4,4)
    e = model(H, t).squeeze()

    mask = e < -78 # emperically set threshold, changes with different ckpt.
    H = H[mask].reshape(-1,4,4)

    print(f"Generated {H.shape[0]} valid grasps")

    H[..., :3, -1] *=1/8.

    ## Visualize results ##
    from se3dif.visualization import grasp_visualization

    vis_H = H.squeeze()
    P *=1/8
    mesh = mesh.apply_scale(1/8)

    grasp_visualization.visualize_grasps(to_numpy(H), p_cloud=P, mesh=mesh, da2=True, save=True)
