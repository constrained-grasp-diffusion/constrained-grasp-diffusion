import copy
import configargparse

import scipy.spatial.transform
import numpy as np
from se3dif.models.loader import load_model
from se3dif.samplers import PartGuidedGrasp_AnnealedLD
from se3dif.utils import to_numpy, to_torch
from se3dif.utils.torch_utils import seed_everything

import torch
import os
import trimesh
import open3d as o3d
import random

from se3dif.visualization import grasp_visualization


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    print()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def parse_args():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--n_grasps', type=str, default='200')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--model', type=str, default='cgdf_v1')
    p.add_argument('--input', type=str, required=True)

    opt = p.parse_args()
    return opt


def get_approximated_grasp_diffusion_field(p, p_, args, device='cpu'):
    model_params = args.model
    batch = int(args.n_grasps)
    ## Load model
    model_args = {
        'device': device,
        'pretrained_model': model_params
    }
    model1 = load_model(model_args)
    model2 = load_model(model_args)

    context = to_torch(p[None,...], device)
    model1.set_latent(context, batch=batch)

    context = to_torch(p_[None,...], device)
    model2.set_latent(context, batch=batch)

    ########### 2. SET SAMPLING METHOD #############
    generator = PartGuidedGrasp_AnnealedLD(model1, model2, batch=batch, T=70, T_fit=50, k_steps=2, device=device)

    return generator, model1, model2


def mean_center_and_normalize(mesh):
    mean = mesh.sample(1000).mean(0)
    mesh.vertices -= mean
    scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
    mesh.apply_scale(1/(1.5*scale))
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

def get_random_target_region(P_):
    indices = knn(to_torch(P_).unsqueeze(0).transpose(1,2), 100)
    index = random.randint(0, P_.shape[0]-1)
    P = P_[indices[0, index]]
    
    pcd = o3d.geometry.PointCloud()
    colors1 = torch.Tensor([[1,0,0]]).repeat(P_.shape[0],1)
    colors2 = torch.Tensor([[0,1,0]]).repeat(P.shape[0],1)
    colors = torch.cat([colors1,colors2],dim=0)
    points = torch.cat([to_torch(P_),to_torch(P)],dim=0)
    pcd.points = o3d.utility.Vector3dVector(points.numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors.numpy())
    
    o3d.io.write_point_cloud('close_region.ply', pcd)
    
    return P


if __name__ == '__main__':

    args = parse_args()
    seed_everything(1234)

    n_grasps = int(args.n_grasps)
    device = args.device
    input_path = args.input

    ## Set Model and Sample Generator ##
    P_, mesh = sample_pointcloud(input_path)
    P = get_random_target_region(P_)
    generator, model1, model2 = get_approximated_grasp_diffusion_field(P, P_, args, device)
    # print(generator)
    # print(model)

    # Running the model
    H_, t = generator.sample()

    H = H_.reshape(-1,4,4)
    e = model2(H, t).squeeze()

    mask = e < -78 # emperically set threshold, changes with different ckpt.
    H = H[mask].reshape(-1,4,4)

    print(f"Generated {H.shape[0]} valid grasps")

    H[..., :3, -1] *=1/8.

    ## Visualize results ##

    vis_H = H.squeeze()
    P *=1/8
    mesh = mesh.apply_scale(1/8)

    grasp_visualization.visualize_grasps(to_numpy(H), p_cloud=P, mesh=mesh, da2=True, save=True)
