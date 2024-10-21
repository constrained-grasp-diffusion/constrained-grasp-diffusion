import glob
import copy
import time

import numpy as np
import trimesh

from scipy.stats import special_ortho_group

import os
import torch

from torch.utils.data import DataLoader, Dataset
import json
import pickle
import h5py
from se3dif.utils import get_data_src

from se3dif.utils import to_numpy, to_torch, get_grasps_src
from mesh_to_sdf.surface_point_cloud import get_scan_view, get_hq_scan_view
from mesh_to_sdf.scan import ScanPointcloud



import os, sys

import logging
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)


class DA2Grasps():
    def __init__(self, filename, single_arm = True):
        self.filename = filename
        scale = None
        if filename.endswith(".json"):
            data = json.load(open(filename, "r"))
            self.mesh_fname = data["object"].decode('utf-8')
            self.mesh_type = self.mesh_fname.split('/')[1]
            self.mesh_id = self.mesh_fname.split('/')[-1].split('.')[0]
            self.mesh_scale = data["object_scale"] if scale is None else scale
        elif filename.endswith(".h5"):
            data = h5py.File(filename, "r")
            self.mesh_fname = 'meshes/' + data["object/file"][()].decode('utf-8')
            # self.mesh_type = self.mesh_fname.split('/')[1]
            self.mesh_id = self.mesh_fname.split('/')[-1].split('.')[0]
            self.mesh_scale = data["object/scale"][()] if scale is None else scale
        else:
            raise RuntimeError("Unknown file ending:", filename)

        self.grasps, self.success = self.load_grasps(filename)
        # dual_negatives_path = os.path.join('/raid/t1/scratch/grasp_dif/da2-positive-neg-1024/',os.path.basename(filename)[:-3]+'.pickle')
        # neg_grasp_file = pickle.load(open(dual_negatives_path,'rb'))
        # self.ng_index = np.array(neg_grasp_file['negative_pairs'])

        self.quality_to_class()

        good_idxs = np.argwhere(self.success>=0.5)[:,0]
        bad_idxs  = np.argwhere(self.success<0.5)[:,0]
        self.good_grasps = self.grasps[good_idxs,...]
        self.bad_grasps  = self.grasps[bad_idxs,...]
        if single_arm:
            self.good_grasps = self.good_grasps.reshape(-1,4,4)
            self.bad_grasps = self.bad_grasps.reshape(-1,4,4)
    
    def quality_to_class(self):
        bin_boundaries = [0, 0.85, 1+1e-5]
        self.bin_indices = np.digitize(self.success, bin_boundaries) - 1

    def load_grasps(self, filename):
        """Load transformations and qualities of grasps from a JSON file from the dataset.

        Args:
            filename (str): HDF5 or JSON file name.

        Returns:
            np.ndarray: Homogenous matrices describing the grasp poses. 2000 x 4 x 4.
            np.ndarray: List of binary values indicating grasp success in simulation.
        """
        if filename.endswith(".json"):
            data = json.load(open(filename, "r"))
            T = np.array(data["transforms"])
            success = np.array(data["quality_flex_object_in_gripper"])
        elif filename.endswith(".h5"):
            data = h5py.File(filename, "r")
            T = np.array(data["grasps/transforms"])
            success = \
                0.5 * np.array(data["grasps/qualities/Force_closure"]) + \
                0.4 * np.array(data["grasps/qualities/Dexterity"]) + \
                0.1 * np.array(data["grasps/qualities/Torque_optimization"])
        else:
            raise RuntimeError("Unknown file ending:", filename)
        return T, success
    
    def load_mesh(self):
        mesh_path_file = os.path.join(get_data_src(), self.mesh_fname)

        mesh = trimesh.load(mesh_path_file,  file_type='obj', force='mesh')

        mesh.apply_scale(self.mesh_scale)
        if type(mesh) == trimesh.scene.scene.Scene:
            mesh = trimesh.util.concatenate(mesh.dump())
        return mesh


class DA2GraspsDirectory():
    def __init__(self, filename=get_grasps_src(), data_type='Mug', single_arm=True):

        self.grasps_files = sorted(glob.glob(filename + '/*.h5'))

        self.avail_obj = []
        for grasp_file in self.grasps_files:
            self.avail_obj.append(DA2Grasps(grasp_file, single_arm=single_arm))


class DA2AndSDFDataset(Dataset):
    'DataLoader for training DeepSDF Auto-Decoder model'
    def __init__(self, class_type='Mug', se3=False, phase='train', one_object=False,
                 n_pointcloud = 1000, n_density = 200, n_coords = 1500,
                 augmented_rotation=True, visualize=False, split = True , exp_feature=False):

        self.class_type = class_type
        self.data_dir = get_data_src()
        self.DA2_data_dir = self.data_dir

        self.grasps_dir = os.path.join(self.DA2_data_dir, 'grasps')
        self.sdf_dir = os.path.join(self.DA2_data_dir, 'sdf')

        self.generated_points_dir = os.path.join(self.DA2_data_dir, 'train_data')
        
        self.exp_feature = exp_feature
        
        grasps_files = sorted(glob.glob(self.grasps_dir+'/'+class_type+'/*.h5'))
        

        points_files = []
        sdf_files = []
        for grasp_file in grasps_files:
            g_obj = DA2Grasps(grasp_file)
            mesh_file = g_obj.mesh_fname
            txt_split = mesh_file.split('/')

            sdf_file = os.path.join(self.sdf_dir, class_type, txt_split[-1].split('.')[0]+'.json')
            point_file = os.path.join(self.generated_points_dir, class_type, '4_points', txt_split[-1]+'.npz')

            sdf_files.append(sdf_file)
            points_files.append(point_file)

        ## Split Train/Validation
        n = len(grasps_files)
        indexes = np.arange(0, n)
        self.total_len = n
        if split:
            idx = int(0.9 * n)
        else:
            idx = int(n)

        if phase == 'train':
            self.grasp_files = grasps_files[:idx]
            self.points_files = points_files[:idx]
            self.sdf_files = sdf_files[:idx]
            self.indexes = indexes[:idx]
        else:
            self.grasp_files = grasps_files[idx:]
            self.points_files = points_files[idx:]
            self.sdf_files = sdf_files[idx:]
            self.indexes = indexes[idx:]


        self.len = len(self.points_files)

        self.n_pointcloud = n_pointcloud
        self.n_density  = n_density
        self.n_occ = n_coords

        ## Variables on Data
        self.one_object = one_object
        self.augmented_rotation = augmented_rotation
        self.se3 = se3

        ## Visualization
        self.visualize = visualize
        self.scale = 8.

    def __len__(self):
        return self.len

    def _get_item(self, index):
        if self.one_object:
            index = 0

        index_obj = self.indexes[index]
        ## Load Files ##
        grasps_obj = DA2Grasps(self.grasp_files[index])
        sdf_file = self.sdf_files[index]
        with open(sdf_file, 'rb') as handle:
            sdf_dict = pickle.load(handle)

        ## PointCloud
        p_clouds = sdf_dict['pcl']
        rix = np.random.permutation(p_clouds.shape[0])
        p_clouds = p_clouds[rix[:self.n_pointcloud],:]

        ## Coordinates XYZ
        coords  = sdf_dict['xyz']
        rix = np.random.permutation(coords.shape[0])
        coords = coords[rix[:self.n_occ],:]

        ### SDF value
        sdf = sdf_dict['sdf'][rix[:self.n_occ]]
        grad_sdf = sdf_dict['grad_sdf'][rix[:self.n_occ], ...]

        ### Scale and Loc
        scale = sdf_dict['scale']
        loc = sdf_dict['loc']

        ## Grasps good/bad
        rix = np.random.randint(low=0, high=grasps_obj.good_grasps.shape[0], size=self.n_density)
        H_grasps = grasps_obj.good_grasps[rix, ...]
        rix = np.random.randint(low=0, high=grasps_obj.bad_grasps.shape[0], size=self.n_density)
        H_bad_grasps = grasps_obj.bad_grasps[rix, ...]

        ## Rescale Pointcloud and Occupancy Points ##
        coords = (coords + loc)*scale*grasps_obj.mesh_scale * self.scale
        p_clouds = (p_clouds + loc)*scale*grasps_obj.mesh_scale * self.scale

        sdf = sdf*scale*grasps_obj.mesh_scale * self.scale
        grad_sdf = -grad_sdf*scale*grasps_obj.mesh_scale * self.scale

        H_grasps[:,:-1,-1] = H_grasps[:,:-1,-1] * self.scale
        H_bad_grasps[:,:-1,-1] = H_bad_grasps[:,:-1,-1]*self.scale

        ## Random rotation ##
        if self.augmented_rotation:
            R = special_ortho_group.rvs(3)
            H = np.eye(4)
            H[:3,:3] = R

            coords = np.einsum('mn,bn->bm',R, coords)
            p_clouds = np.einsum('mn,bn->bm',R, p_clouds)

            H_grasps = np.einsum('mn,bnd->bmd', H, H_grasps)
            H_bad_grasps = np.einsum('mn,bnd->bmd', H, H_bad_grasps)

            grad_sdf = np.einsum('mn,bn->bm', R, grad_sdf)


        # Visualize
        if self.visualize:
            ## 3D matplotlib ##
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(p_clouds[:,0], p_clouds[:,1], p_clouds[:,2], c='r')

            n = 10
            x = coords[:n,:]
            ## grad sdf ##
            x_grad = grad_sdf[:n, :]
            ax.quiver(x[:,0], x[:,1], x[:,2], x_grad[:,0], x_grad[:,1], x_grad[:,2], length=0.3)

            ## sdf visualization ##
            x_sdf = sdf[:n]
            x_sdf = 0.9*x_sdf/np.max(x_sdf)
            c = np.zeros((n, 3))
            c[:, 1] = x_sdf
            ax.scatter(x[:,0], x[:,1], x[:,2], c=c)

            plt.show(block=True)

        del sdf_dict
        
        
        if self.exp_feature:
            num_of_grasps = 200
            dist_threshold = 0.6 
            gripper_pos = np.expand_dims(H_grasps[:num_of_grasps,:3,3],axis=0) # [num_of_grasps,3]
            gripper_pos = np.expand_dims(gripper_pos,axis=1) # [num_of_grasps,3] --> [num_of_grasps,1,3] 
            
            pts = np.expand_dims(p_clouds,axis=0) # [num_of_pts,3] --> [1,num_of_pts,3]
            diff_btw_g2pt = gripper_pos - pts # [num_of_grasps,num_of_pts,3]
            
            dist_map = np.sqrt(np.expand_dims((diff_btw_g2pt**2).sum(axis=2),axis=-1)) # [num_of_grasp,num_of_pts,1] euclidean metric
            
            normalized_dist_map = (dist_map-dist_map.min(axis=0))/(dist_map.max(axis=0)-dist_map.min(axis=0)) # [num_of_grasp,num_of_pts,1] value in range [0,1].
            normalized_dist_map = np.abs(normalized_dist_map - 1.) # np.abs(nomalized_dist_map -1.) this is now a closeness map in range [0,1] i.e. close pts wrt to grasp will have value close to 1.
            
            closeness_mask = np.zeros_like(normalized_dist_map)
            closeness_mask[np.where(normalized_dist_map>dist_threshold)] = 1.
            
            res = {'point_cloud': torch.from_numpy(p_clouds).float(),
               'x_sdf': torch.from_numpy(coords).float(),
               'x_ene_pos': torch.from_numpy(H_grasps).float(),
               'x_neg_ene': torch.from_numpy(H_bad_grasps).float(),
               'scale': torch.Tensor([self.scale]).float(),
               'visual_context':  torch.Tensor([index_obj]),
               'closeness_score': torch.from_numpy(normalized_dist_map).float(),
               'closeness_mask': torch.from_numpy(closeness_mask).float()}
        else:
            res = {'point_cloud': torch.from_numpy(p_clouds).float(),
                'x_sdf': torch.from_numpy(coords).float(),
                'x_ene_pos': torch.from_numpy(H_grasps).float(),
                'x_neg_ene': torch.from_numpy(H_bad_grasps).float(),
                'scale': torch.Tensor([self.scale]).float(),
                'visual_context':  torch.Tensor([index_obj])}

        return res, {'sdf': torch.from_numpy(sdf).float(), 'grad_sdf': torch.from_numpy(grad_sdf).float()}

    def __getitem__(self, index):
        'Generates one sample of data'
        return self._get_item(index)


class PointcloudDA2AndSDFDataset(Dataset):
    'DataLoader for training DeepSDF with a Rotation Invariant Encoder model'
    def __init__(self, class_type=['Cup', 'Mug', 'Fork', 'Hat', 'Bottle', 'Bowl', 'Car', 'Donut', 'Laptop', 'MousePad', 'Pencil',
                                   'Plate', 'ScrewDriver', 'WineBottle','Backpack', 'Bag', 'Banana', 'Battery', 'BeanBag', 'Bear',
                                   'Book', 'Books', 'Camera','CerealBox', 'Cookie','Hammer', 'Hanger', 'Knife', 'MilkCarton', 'Painting',
                                   'PillBottle', 'Plant','PowerSocket', 'PowerStrip', 'PS3', 'PSP', 'Ring', 'Scissors', 'Shampoo', 'Shoes',
                                   'Sheep', 'Shower', 'Sink', 'SoapBottle', 'SodaCan','Spoon', 'Statue', 'Teacup', 'Teapot', 'ToiletPaper',
                                   'ToyFigure', 'Wallet','WineGlass',
                                   'Cow', 'Sheep', 'Cat', 'Dog', 'Pizza', 'Elephant', 'Donkey', 'RubiksCube', 'Tank', 'Truck', 'USBStick'],
                 se3=False, phase='train', one_object=False,
                 n_pointcloud = 1000, n_density = 200, n_coords = 1000,
                 augmented_rotation=True, visualize=False, single_arm=True, split = True, exp_feature=False):

        #class_type = ['Mug']
        self.class_type = class_type
        self.single_arm = single_arm
        self.exp_feature = exp_feature
        self.data_dir = get_data_src()

        self.grasps_dir = os.path.join(self.data_dir, 'grasps')

        self.grasp_files = []
        # for class_type_i in class_type:
        cls_grasps_files = sorted(glob.glob(self.grasps_dir+'/*.h5'))

        for grasp_file in cls_grasps_files:
            g_obj = DA2Grasps(grasp_file, single_arm=single_arm)

            ## Grasp File ##
            if g_obj.good_grasps.shape[0] > 0:
                self.grasp_files.append(grasp_file)

        ## Split Train/Validation
        n = len(self.grasp_files)
        train_size = int(n*0.9)
        test_size  =  n - train_size

        self.train_grasp_files, self.test_grasp_files = torch.utils.data.random_split(self.grasp_files, [train_size, test_size])
        
        # print(self.train_grasp_files.indices)
        print(self.test_grasp_files.indices)
        # exit()

        self.type = 'train'
        self.len = len(self.train_grasp_files)

        self.n_pointcloud = n_pointcloud
        self.n_density  = n_density
        self.n_occ = n_coords

        ## Variables on Data
        self.one_object = one_object
        self.augmented_rotation = augmented_rotation
        self.se3 = se3

        ## Visualization
        self.visualize = visualize
        self.scale = 8.

    def __len__(self):
        return self.len

    def set_test_data(self):
        self.len = len(self.test_grasp_files)
        self.type = 'test'

    def _get_grasps(self, grasp_obj):
        try:
            rix = np.random.randint(low=0, high=grasp_obj.good_grasps.shape[0], size=self.n_density)
        except:
            print('lets see')
        H_grasps = grasp_obj.good_grasps[rix, ...]
        return H_grasps

    def _get_sdf(self, grasp_obj, grasp_file):

        mesh_fname = grasp_obj.mesh_fname
        mesh_scale = grasp_obj.mesh_scale

        mesh_name = mesh_fname.split('/')[-1]
        filename  = mesh_name.split('.obj')[0]
        sdf_file = os.path.join(self.data_dir, 'sdf', filename+'.json')

        with open(sdf_file, 'rb') as handle:
            sdf_dict = pickle.load(handle)

        loc = sdf_dict['loc']
        scale = sdf_dict['scale']
        xyz = (sdf_dict['xyz'] + loc)*scale*mesh_scale
        rix = np.random.permutation(xyz.shape[0])
        xyz = xyz[rix[:self.n_occ], :]
        sdf = sdf_dict['sdf'][rix[:self.n_occ]]*scale*mesh_scale
        return xyz, sdf

    def _get_mesh_pcl(self, grasp_obj):
        mesh = grasp_obj.load_mesh()
        return mesh.sample(self.n_pointcloud)

    def _get_grasps_qualities(self, grasp_obj):
        try:
            rix = np.random.randint(low=0, high=grasp_obj.grasps.shape[0], size=self.n_density * 3 // 4)
        except:
            print('lets see')
        H_grasps = grasp_obj.grasps[rix, ...]
        score = grasp_obj.bin_indices[rix, ...]
        # hard_negative_grasps = grasp_obj.ng_index
        # H_negative_grasps = grasp_obj.grasps.reshape(-1,4,4)[hard_negative_grasps, ...]
        # try:
        #     rix = np.random.randint(low=0, high=H_negative_grasps.shape[0], size=self.n_density // 4)
        # except:
        #     print('lets see')
        # H_negative_grasps = H_negative_grasps[rix, ...]
        # hard_ng_scores = np.zeros((H_negative_grasps.shape[0],1)).astype(np.int64)
        # H_grasps = np.concatenate([H_grasps,H_negative_grasps],axis=0)
        # score = np.concatenate([score,hard_ng_scores],axis=0)
        
        # print(H_grasps.shape, score.shape)
        # exit()
        
        return H_grasps, score

    def _get_item(self, index):
        if self.one_object:
            index = 0

        ## Load Files ##
        if self.type == 'train':
            grasps_obj = DA2Grasps(self.train_grasp_files[index], single_arm=self.single_arm)
        else:
            grasps_obj = DA2Grasps(self.test_grasp_files[index], single_arm=self.single_arm)

        ## SDF
        xyz, sdf = self._get_sdf(grasps_obj, self.train_grasp_files[index])

        ## PointCloud
        pcl = self._get_mesh_pcl(grasps_obj)

        ## Grasps good/bad
        H_grasps = self._get_grasps(grasps_obj)

        ## Quality metrics
        qual_g, score = self._get_grasps_qualities(grasps_obj)
        ## rescale, rotate and translate ##
        xyz = xyz*self.scale
        sdf = sdf*self.scale
        pcl = pcl*self.scale
        H_grasps[..., :3, -1] = H_grasps[..., :3, -1]*self.scale
        qual_g[..., :3, -1] = qual_g[..., :3, -1]*self.scale
        ## Random rotation ##
        R = special_ortho_group.rvs(3)
        H = np.eye(4)
        H[:3, :3] = R
        mean = np.mean(pcl, 0)
        ## translate ##
        xyz = xyz - mean
        pcl = pcl - mean
        H_grasps[..., :3, -1] = H_grasps[..., :3, -1] #- mean
        ## rotate ##
        pcl = np.einsum('mn,bn->bm',R, pcl)
        xyz = np.einsum('mn,bn->bm',R, xyz)
        if self.single_arm:
            H_grasps = np.einsum('mn,bnk->bmk', H, H_grasps)
            qual_g = np.einsum('mn,bnk->bmk', H, qual_g)
        else:
            H_grasps = np.einsum('mn,bonk->bomk', H, H_grasps)
            qual_g = np.einsum('mn,bonk->bomk', H, qual_g)
        #######################

        # Visualize
        if self.visualize:

            # ## 3D matplotlib ##
            # import matplotlib.pyplot as plt

            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # ax.scatter(pcl[:,0], pcl[:,1], pcl[:,2], c='r')

            # x_grasps = H_grasps[..., :3, -1].reshape(-1,3)
            # ax.scatter(x_grasps[:,0], x_grasps[:,1], x_grasps[:,2], c='b')

            # ## sdf visualization ##
            # n = 100
            # x = xyz[:n,:]

            # x_sdf = sdf[:n]
            # x_sdf = 0.9*x_sdf/np.max(x_sdf)
            # c = np.zeros((n, 3))
            # c[:, 1] = x_sdf
            # ax.scatter(x[:,0], x[:,1], x[:,2], c=c)
            # np.savez('sdf_viz.npz', pcl=pcl,x_grasps=x_grasps,c=c,x=x)
            # exit()
            # plt.savefig('vis.png')
            # exit()

            # plt.show()
            #plt.show(block=True)
            from se3dif.visualization import grasp_visualization
            mesh = grasps_obj.load_mesh().apply_scale(self.scale)
            mesh.vertices -= mean
            mesh = mesh.apply_transform(H)
            # grasp_visualization.visualize_grasps(H_grasps[10], p_cloud=pcl, mesh=mesh,scale=self.scale, da2=True, dual=True)
        
        if self.exp_feature:
            num_of_grasps = 200
            dist_threshold = 0.9
            gripper_pos = H_grasps[...,:3,3] # [n_density, 3] or [n_density,2,3]
            
            gripper_pos = np.expand_dims(gripper_pos,axis=-2) # [n_density,2,3] --> [n_density,2,1,3]
            pts = np.expand_dims(pcl,axis=0) # [num_of_pts,3] --> [1,num_of_pts,3]
            
            if not self.single_arm:
                pts = np.expand_dims(pts, axis=0) # [1,num_of_pts,3] --> [1,1,num_of_pts,3]

            diff_btw_g2pt = gripper_pos - pts # [n_density,2,num_of_pts,3]
            
            dist_map = np.sqrt(np.expand_dims((diff_btw_g2pt**2).sum(axis=-1),axis=-1)) # [num_of_grasp,num_of_pts,2,1] euclidean metric
            
            normalized_dist_map = (dist_map-dist_map.min(axis=0))/(dist_map.max(axis=0)-dist_map.min(axis=0)) # [num_of_grasp,num_of_pts,1] value in range [0,1].
            normalized_dist_map = np.abs(normalized_dist_map - 1.) ** 4# np.abs(nomalized_dist_map -1.) this is now a closeness map in range [0,1] i.e. close pts wrt to grasp will have value close to 1.
            # print(normalized_dist_map.min(), normalized_dist_map.max())
            
            closeness_mask = np.zeros_like(normalized_dist_map)
            closeness_mask[np.where(normalized_dist_map>dist_threshold)] = 1.
            
            res = {
               'x_sdf': torch.from_numpy(xyz).float(),
               'x_ene_pos': torch.from_numpy(H_grasps).float(),
               'scale': torch.Tensor([self.scale]).float(),
               'visual_context': torch.from_numpy(pcl).float(),
               'closeness_score': torch.from_numpy(normalized_dist_map).float(),
               'closeness_mask': torch.from_numpy(closeness_mask).int()}
        else:
            res = {'visual_context': torch.from_numpy(pcl).float(),
                'x_sdf': torch.from_numpy(xyz).float(),
                'x_ene_pos': torch.from_numpy(H_grasps).float(),
                'qual_g': torch.from_numpy(qual_g).float(),
                'scale': torch.Tensor([self.scale]).float()}
            gt = {'sdf': torch.from_numpy(sdf).float(),
                  'score': torch.from_numpy(score).long()
                  }
        
        if self.visualize:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            points = torch.from_numpy(copy.deepcopy(pcl))
            import matplotlib.pyplot as plt
            plt.hist(sdf,bins=200)
            plt.savefig('sdfvals.png')
            # indices = closeness_mask[10,0,:,0] == 1
            # indices2 = closeness_mask[10,0,:,0] == 0
            # points1 = points[indices]
            # points2 = points[indices2]
            # print(points1.shape, points2.shape)
            # colors1 = torch.Tensor([[1,0,0]]).repeat(points1.shape[0], 1)
            # colors2 = torch.Tensor([[0,1,0]]).repeat(points2.shape[0], 1)
            # colors = torch.cat([colors1, colors2],dim=0)
            # points = torch.cat([points1,points2],dim=0)
            colors = torch.Tensor([[1,1,0]]).repeat(points.shape[0], 1)
            colors[...,0] *= normalized_dist_map[42,0,:,0]
            colors[...,1] *= 1 - normalized_dist_map[42,0,:,0]
            # colors = torch.cat([colors,torch.ones_like(torch.from_numpy(xyz))],dim=0)
            # points = torch.cat([points,torch.from_numpy(xyz)],dim=0)
            pcd.points = o3d.utility.Vector3dVector(points.numpy())
            pcd.colors = o3d.utility.Vector3dVector(colors.numpy())
            o3d.io.write_point_cloud('close_region.ply', pcd)
            grasp_visualization.visualize_grasps(H_grasps[42], scale=self.scale, da2=True)
            

        return res, gt

    def __getitem__(self, index):
        'Generates one sample of data'
        # index = 1212
        # print(index)
        return self._get_item(index)


class PartialPointcloudDA2AndSDFDataset(Dataset):
    'DataLoader for training DeepSDF with a Rotation Invariant Encoder model'
    def __init__(self, class_type=['Cup', 'Mug', 'Fork', 'Hat', 'Bottle'],
                 se3=False, phase='train', one_object=False,
                 n_pointcloud = 1000, n_density = 200, n_coords = 1000, augmented_rotation=True, visualize=False, single_arm=True):

        #class_type = ['Mug']
        self.class_type = class_type
        self.single_arm = single_arm
        self.data_dir = get_data_src()

        self.grasps_dir = os.path.join(self.data_dir, 'grasps')

        self.grasp_files = []
        # for class_type_i in class_type:
        cls_grasps_files = sorted(glob.glob(self.grasps_dir+'/*.h5'))

        for grasp_file in cls_grasps_files:
            g_obj = DA2Grasps(grasp_file, single_arm=single_arm)

            ## Grasp File ##
            if g_obj.good_grasps.shape[0] > 0:
                self.grasp_files.append(grasp_file)

        ## Split Train/Validation
        n = len(self.grasp_files)
        train_size = int(n*0.9)
        test_size  =  n - train_size

        self.train_grasp_files, self.test_grasp_files = torch.utils.data.random_split(self.grasp_files, [train_size, test_size])
        
        print(len(self.train_grasp_files))
        print(self.test_grasp_files.indices)
        # exit()

        self.type = 'train'
        self.len = len(self.train_grasp_files)

        self.n_pointcloud = n_pointcloud
        self.n_density  = n_density
        self.n_occ = n_coords

        ## Variables on Data
        self.one_object = one_object
        self.augmented_rotation = augmented_rotation
        self.se3 = se3

        ## Visualization
        self.visualize = visualize
        self.scale = 8.

        ## Sampler
        self.scan_pointcloud = None

    def __len__(self):
        return self.len

    def set_test_data(self):
        self.len = len(self.test_grasp_files)
        self.type = 'test'

    def _get_grasps(self, grasp_obj):
        try:
            rix = np.random.randint(low=0, high=grasp_obj.good_grasps.shape[0], size=self.n_density)
        except:
            print('lets see')
        H_grasps = grasp_obj.good_grasps[rix, ...]
        return H_grasps

    def _get_sdf(self, grasp_obj, grasp_file):

        mesh_fname = grasp_obj.mesh_fname
        mesh_scale = grasp_obj.mesh_scale

        mesh_name = mesh_fname.split('/')[-1]
        filename  = mesh_name.split('.obj')[0]
        sdf_file = os.path.join(self.data_dir, 'sdf', filename+'.json')

        with open(sdf_file, 'rb') as handle:
            sdf_dict = pickle.load(handle)

        loc = sdf_dict['loc']
        scale = sdf_dict['scale']
        xyz = (sdf_dict['xyz'] + loc)*scale*mesh_scale
        rix = np.random.permutation(xyz.shape[0])
        xyz = xyz[rix[:self.n_occ], :]
        sdf = sdf_dict['sdf'][rix[:self.n_occ]]*scale*mesh_scale
        return xyz, sdf


    def _get_mesh_pcl(self, grasp_obj):
        mesh = grasp_obj.load_mesh()
        ## 1. Mesh Centroid ##
        centroid = mesh.centroid
        H = np.eye(4)
        H[:3, -1] = -centroid
        mesh.apply_transform(H)
        ######################
        #time0 = time.time()
        P = self.scan_pointcloud.get_hq_scan_view(mesh)
        #print('Sample takes {} s'.format(time.time() - time0))
        P +=centroid
        try:
            rix = np.random.randint(low=0, high=P.shape[0], size=self.n_pointcloud)
        except:
            print('here')
        return P[rix, :]

    def _get_item(self, index):
        if self.one_object:
            index = 0

        ## Load Files ##
        if self.type == 'train':
            grasps_obj = DA2Grasps(self.train_grasp_files[index], single_arm=self.single_arm)
        else:
            grasps_obj = DA2Grasps(self.test_grasp_files[index], single_arm=self.single_arm)

        ## SDF
        xyz, sdf = self._get_sdf(grasps_obj, self.train_grasp_files[index])

        ## PointCloud
        pcl = self._get_mesh_pcl(grasps_obj)

        ## Grasps good/bad
        H_grasps = self._get_grasps(grasps_obj)

        ## rescale, rotate and translate ##
        xyz = xyz*self.scale
        sdf = sdf*self.scale
        pcl = pcl*self.scale
        H_grasps[..., :3, -1] = H_grasps[..., :3, -1]*self.scale
        ## Random rotation ##
        R = special_ortho_group.rvs(3)
        H = np.eye(4)
        H[:3, :3] = R
        mean = np.mean(pcl, 0)
        ## translate ##
        xyz = xyz - mean
        pcl = pcl - mean
        H_grasps[..., :3, -1] = H_grasps[..., :3, -1]
        ## rotate ##
        pcl = np.einsum('mn,bn->bm',R, pcl)
        xyz = np.einsum('mn,bn->bm',R, xyz)
        if self.single_arm:
            H_grasps = np.einsum('mn,bnk->bmk', H, H_grasps)
            # qual_g = np.einsum('mn,bnk->bmk', H, qual_g)
        else:
            H_grasps = np.einsum('mn,bonk->bomk', H, H_grasps)
            # qual_g = np.einsum('mn,bonk->bomk', H, qual_g)
        #######################

        # Visualize
        if self.visualize:

            ## 3D matplotlib ##
            import matplotlib.pyplot as plt

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(pcl[:,0], pcl[:,1], pcl[:,2], c='r')

            x_grasps = H_grasps[..., :3, -1]
            ax.scatter(x_grasps[:,0], x_grasps[:,1], x_grasps[:,2], c='b')

            ## sdf visualization ##
            n = 100
            x = xyz[:n,:]

            x_sdf = sdf[:n]
            x_sdf = 0.9*x_sdf/np.max(x_sdf)
            c = np.zeros((n, 3))
            c[:, 1] = x_sdf
            ax.scatter(x[:,0], x[:,1], x[:,2], c=c)

            plt.show()
            #plt.show(block=True)

        res = {'visual_context': torch.from_numpy(pcl).float(),
               'x_sdf': torch.from_numpy(xyz).float(),
               'x_ene_pos': torch.from_numpy(H_grasps).float(),
               'scale': torch.Tensor([self.scale]).float()}
        # for key in res.keys():
        #     print(key, res[key].shape)
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(to_numpy(res['visual_context']))
        # o3d.io.write_point_cloud('partial_test.pcd', pcd)
        # exit()

        return res, {'sdf': torch.from_numpy(sdf).float()}

    def __getitem__(self, index):
        'Generates one sample of data'
        return self._get_item(index)


if __name__ == '__main__':
    from se3dif.utils.torch_utils import seed_everything
    seed_everything()
    ## Index conditioned dataset
    # dataset = DA2AndSDFDataset(visualize=True, augmented_rotation=True, one_object=False)

    ## Pointcloud conditioned dataset
    # dataset = PointcloudDA2AndSDFDataset(visualize=True, augmented_rotation=True, one_object=False, single_arm=False, exp_feature=True, n_pointcloud=4096)

    dataset = DA2GraspsDirectory(single_arm=False)
    print(dataset.avail_obj[0])
