#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrix2

class Camera(nn.Module):
    def __init__(
        self, 
        id,
        R, T, 
        FoVx, FoVy, K,
        image, image_name, 
        trans = np.array([0.0, 0.0, 0.0]), 
        scale = 1.0,
        metadata = dict(),
        masks = dict(),

        # extend
        depth=None, 
        gt_alpha_mask=None,
        data_device=None, time=None,
        Znear=None, Zfar=None, 
        h=None, w=None,
        mask = None,
        colmap_id = None,
        uid = None,
        dataset_name_for_different_z = None,
        depth_sudden_change_mask = None,
    ):
        super(Camera, self).__init__()

        #shared
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.trans, self.scale = trans, scale
        self.dataset_name_for_different_z = dataset_name_for_different_z

        #exlusive to deform3dgs
        self.uid = uid
        self.colmap_id = colmap_id
        self.time = time
        self.mask = mask

        # exclusive for ttgs
        self.K = K


        # if dataset == 'EndoNeRF':
        #     K = np.array([[569.4682,   0.  ,    320.     ],
        #                     [  0.   ,   569.4682, 256.     ],
        #                     [  0.   ,    0.  ,      1.     ]] )
        # elif dataset == 'SM':
        #     K = np.array([[560.0158,   0.  ,    320.     ],
        #                     [  0.   ,   560.01587, 256.     ],
        #                     [  0.   ,    0.  ,      1.     ]] )
        # else:
        #     assert 0,dataset

        # assert 0,self.K
        self.meta = metadata
        self.id = id

        self.masks_dict_keys = list(masks.keys())
        for name, mask_i in masks.items():
            setattr(self, name, mask_i)
        if 'ego_pose' in self.meta.keys():
            self.ego_pose = torch.from_numpy(self.meta['ego_pose']).float().cuda()
            del self.meta['ego_pose']
            
        if 'extrinsic' in self.meta.keys():
            self.extrinsic = torch.from_numpy(self.meta['extrinsic']).float().cuda()
            del self.meta['extrinsic']

        #again shared
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        
        self.original_image = image.clamp(0.0, 1.0)
        self.original_depth = depth
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width))
        # extend        
        self.depth_sudden_change_mask = depth_sudden_change_mask

        # assert 0,masks.keys()

        # debug_bad_bg = True
        # if debug_bad_bg:
        #     mask_key = 'raw_obj_tool1'
        #     self.original_image *= masks[mask_key]

        #     self.original_image = torch.ones_like(self.original_image).to(self.original_image.device)
        
        if Zfar is not None and Znear is not None:
            self.zfar = Zfar
            self.znear = Znear
        else:
            if self.dataset_name_for_different_z in ['EndoNeRF']:
                # ENDONERF
                self.zfar = 120.0
                self.znear = 0.01
            elif self.dataset_name_for_different_z in ['StereoMIS']:
                # StereoMIS
                self.zfar = 250
                self.znear= 0.03
            else:
                assert 0 ,self.dataset_name_for_different_z

            # streetgs waymo
            # self.zfar = 1000.0
            # self.znear = 0.001

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        if K is None or h is None or w is None:
            assert 0
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        else:
            self.projection_matrix = getProjectionMatrix2(znear=self.znear, zfar=self.zfar, K=K, h = h, w=w).transpose(0,1)
        
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]




        # self.original_image = image.clamp(0, 1)                
        # self.image_height, self.image_width = self.original_image.shape[1], self.original_image.shape[2]
        # self.zfar = 1000.0
        # self.znear = 0.001
        # self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        
        # if self.K is not None:
        #     self.projection_matrix = getProjectionMatrixK(znear=self.znear, zfar=self.zfar, K=self.K, H=self.image_height, W=self.image_width).transpose(0,1).cuda()
        #     self.K = torch.from_numpy(self.K).float().cuda()
        # else:
        #     self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()

        # self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # self.camera_center = self.world_view_transform.inverse()[3, :3]
        



        #load for shared with deform3dgs

                
    def set_extrinsic(self, c2w):
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3].T
        T = w2c[:3, 3]
        
        # set R, T
        self.R = R
        self.T = T
        
        # change attributes associated with R, T
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def set_intrinsic(self, K):
        self.K = torch.from_numpy(K).float().cuda()
        self.projection_matrix = getProjectionMatrixK(znear=self.znear, zfar=self.zfar, K=self.K, H=self.image_height, W=self.image_width).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
    
    def get_extrinsic(self):
        w2c = np.eye(4)
        w2c[:3, :3] = self.R.T
        w2c[:3, 3] = self.T
        c2w = np.linalg.inv(w2c)
        return c2w
    
    def get_intrinsic(self):
        ixt = self.K.cpu().numpy()
        return ixt

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, time):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.time = time

