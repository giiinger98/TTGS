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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.flexible_deform_model import TissueGaussianModel
from scene.tool_model import ToolModel
from arguments import ModelParams

from typing import Union
# from gaussian_model_base import GaussianModelBase
from scene.tt_gaussian_model import TTGaussianModel
from config.argsgroup_ttgs import ModParams
import torch.nn as nn


class Scene:

    # gaussians : TissueGaussianModel
    # gaussians : Union[GaussianModelBase, TTGaussianModel]
    gaussians_or_controller : Union[TissueGaussianModel, TTGaussianModel]
    
    def __init__(self, \
                #  args : ModelParams,
                 args : Union[ModelParams,ModParams],#Dataparams used for load scene_meta
                 load_other_obj_meta = False,
                 new_cfg = None,
                 load_pcd_dict_in_sceneinfo = False,
                 ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.load_pcd_dict_in_sceneinfo = load_pcd_dict_in_sceneinfo
        self.model_path = args.model_path

        if os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")) and args.extra_mark == 'endonerf':
            scene_info = sceneLoadTypeCallbacks["endonerf"](args.source_path,
                                                            tool_mask=args.tool_mask,
                                                            tissue_init_mode=args.tissue_init_mode,
                                                            tool_init_mode=args.tool_init_mode,
                                                            load_other_obj_meta=load_other_obj_meta,
                                                            cfg = new_cfg,
                                                            load_pcd_dict_in_sceneinfo=self.load_pcd_dict_in_sceneinfo,
                                                            load_cotrackerPnpPose=args.load_cotrackerPnpPose,
                                                            exp_path = args.model_path,
                                                            init_detail_params_dict = {} if (not load_pcd_dict_in_sceneinfo) and args.tissue_init_mode in ['MAPF','skipMAPF'] else args.init_detail_params_dict,
                                                            process_tissue_mask_init = args.process_tissue_mask_init,
                                                            inited_pcd_noise_removal = args.inited_pcd_noise_removal,
                                                            supervise_depth_noise_ignore_tgt = args.supervise_depth_noise_ignore_tgt,
                                                            )
            print("Found poses_bounds.py and extra marks with EndoNeRf")
        else:
            assert 0, "Could not recognize scene type!"
                
        self.maxtime = scene_info.maxtime

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.cameras_extent = 10
        self.cameras_extent = 0
        self.cameras_extent = args.camera_extent

        print("Loading Training Cameras")
        self.train_camera = scene_info.train_cameras 
        print("Loading Test Cameras")
        self.test_camera = scene_info.test_cameras 
        print("Loading Video Cameras")
        self.video_camera =  scene_info.video_cameras 
        

        self.point_cloud =  scene_info.point_cloud 
        self.point_cloud_dict =  scene_info.point_cloud_dict 

        self.scene_metadata = scene_info.scene_metadata
        self.loaded_obj_names = scene_info.loaded_obj_names

        self.loaded_iter = None
        self.gaussians_or_controller = None



    def gs_init(self,gaussians_or_controller : Union[TissueGaussianModel, TTGaussianModel],\
                load_iteration=None,
                reset_camera_extent = None,
                load_which_pcd = 'point_cloud',
                ):
        self.gaussians_or_controller = gaussians_or_controller
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
        if self.loaded_iter:
            self.gaussians_or_controller.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           f"{load_which_pcd}.ply"))
            self.gaussians_or_controller.load_model(os.path.join(self.model_path,
                                                    f"{load_which_pcd}",
                                                    "iteration_" + str(self.loaded_iter),
                                                   ))
        else:
            if self.load_pcd_dict_in_sceneinfo:
                assert self.point_cloud_dict!={}
                assert isinstance(self.gaussians_or_controller,nn.Module)
                self.gaussians_or_controller.create_from_pcd(pcd_dict=self.point_cloud_dict, 
                                                             spatial_lr_scale=reset_camera_extent, 
                                                             time_line=self.maxtime)
            else:
                assert self.point_cloud!=None
                assert isinstance(self.gaussians_or_controller, TissueGaussianModel) \
                    or isinstance(self.gaussians_or_controller, ToolModel) 
                 #gs model
                self.gaussians_or_controller.create_from_pcd(pcd=self.point_cloud, 
                                                             spatial_lr_scale=reset_camera_extent, 
                                                             time_line=self.maxtime)

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))
        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians_or_controller.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    
    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        return self.test_camera

    def getVideoCameras(self, scale=1.0):
        return self.video_camera
    
    def getSceneMetaData(self):
        return self.scene_metadata
    
