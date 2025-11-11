import warnings

warnings.filterwarnings("ignore")

import json
import os
import random
import os.path as osp

import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple
from utils.graphics_utils import focal2fov, fov2focal
import glob
from torchvision import transforms as T
# import open3d as o3d
from tqdm import trange
import imageio.v2 as iio
import cv2
import copy
import torch
import torch.nn.functional as F
from utils.general_utils import inpaint_depth, inpaint_rgb
from utils.scene_utils import get_vertices_from_min_max_bounds,plot_6d_bbox_with_pts,get_depth_sudden_change_mask
from utils.sh_utils import SH2RGB

def generate_se3_matrix(translation, rotation_rad):


    # Create rotation matrices around x, y, and z axes
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rotation_rad[0]), -np.sin(rotation_rad[0])],
                   [0, np.sin(rotation_rad[0]), np.cos(rotation_rad[0])]])

    Ry = np.array([[np.cos(rotation_rad[1]), 0, np.sin(rotation_rad[1])],
                   [0, 1, 0],
                   [-np.sin(rotation_rad[1]), 0, np.cos(rotation_rad[1])]])

    Rz = np.array([[np.cos(rotation_rad[2]), -np.sin(rotation_rad[2]), 0],
                   [np.sin(rotation_rad[2]), np.cos(rotation_rad[2]), 0],
                   [0, 0, 1]])

    # Combine rotations
    R = np.dot(Rz, np.dot(Ry, Rx))

    # Create S(3) matrix
    se3_matrix = np.eye(4)
    se3_matrix[:3, :3] = R
    se3_matrix[:3, 3] = translation

    return se3_matrix

def update_extr(c2w, rotation_deg, radii_mm):
        rotation_rad = np.radians(rotation_deg)
        translation = np.array([-radii_mm * np.sin(rotation_rad) , 0, radii_mm * (1 - np.cos(rotation_rad))])
        # translation = np.array([0, 0, 10])
        se3_matrix = generate_se3_matrix(translation, (0,rotation_rad,0)) # transform_C_C'
        extr = np.linalg.inv(se3_matrix) @ np.linalg.inv(c2w) # transform_C'_W = transform_C'_C @ (transform_W_C)^-1
        
        return np.linalg.inv(extr) # c2w

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


class EndoNeRF_Dataset(object):
    def __init__(
        self,
        datadir,
        downsample=1.0,
        test_every=8,
        tool_mask = 'use',
        load_cotrackerPnpPose = False,
        init_detail_params_dict = {},
        process_tissue_mask_init = None,
        model_path = None,
        # compute_depth_sudden_change_mask = False,
        supervise_depth_noise_ignore_tgt = [],
    ):
        # used for potentila depth supervise noise removal
        # self.compute_depth_sudden_change_mask = compute_depth_sudden_change_mask
        self.supervise_depth_noise_ignore_tgt = supervise_depth_noise_ignore_tgt

        self.model_path = model_path
        self.process_tissue_mask_init = process_tissue_mask_init
        # used to debug short sequence of a given seqeunce
        self.debug_reduce = True
        self.debug_reduce = False
        # self.reduced_to = 25
        self.reduced_to = 80

        self.img_wh = (
            int(640 / downsample),
            int(512 / downsample),
        )
        self.root_dir = datadir
        self.downsample = downsample 
        self.blender2opencv = np.eye(4)
        self.transform = T.ToTensor()
        self.white_bg = False
        #extend
        self.tool_mask = tool_mask
        if 'pulling' in self.root_dir or 'cutting' in self.root_dir:
            self.dataset = 'EndoNeRF' 
        elif 'P2_' in self.root_dir:
            self.dataset = "StereoMIS"
        else:
            assert 0, self.root_dir

        self.load_meta()
        print(f"Scene_Meta_data+Cam_K_T loaded, total image:{len(self.image_paths)}")
        
        n_frames = len(self.image_paths)
        self.train_idxs = [i for i in range(n_frames) if (i-1) % test_every != 0]
        self.test_idxs = [i for i in range(n_frames) if (i-1) % test_every == 0]
        self.all_idxs = [i for i in range(n_frames)]
        self.video_idxs = self.all_idxs#self.test_idxs #[i for i in range(n_frames)]
         # extend
        self.camera_timestamps = {"0":{"train_timestamps": self.train_idxs,\
                                "test_timestamps": self.test_idxs,
                                "video_timestamps": self.video_idxs,
                                "all_timestamps": self.all_idxs,
                                }}

        self.maxtime = 1.0
        self.load_cotrackerPnpPose = load_cotrackerPnpPose
        if self.load_cotrackerPnpPose:
            self.obj_poses_path_dict = {}
        self.obj_tracklets = None
        # self.learned_obj_tracklets = None
        self.cotrackerpnp_trajectory_cams2w = None
        # self.cotrackerpnp_trajectory_cams2w_corrected = None
        self.init_detail_params_dict = init_detail_params_dict
    def load_corrected_obj_tracklets(self,pt_path):

        assert os.path.exists(pt_path),f'{pt_path}'
        learned_obj_tracklets = {}
        learned_obj_tracklets = torch.load(pt_path)
        # all_obj_tools_learned
        for tool_obj_name in self.obj_poses_path_dict.keys():
            if tool_obj_name == 'objs_all':
                continue
            assert tool_obj_name.startswith('obj_tool') 
            if self.debug_reduce:
                for key,value in learned_obj_tracklets[tool_obj_name].items():
                    learned_obj_tracklets[tool_obj_name][key] = value[:self.reduced_to]
        
        return learned_obj_tracklets
    def load_meta(self):
        """
        Load Scene_Meta_data+Cam_K_T loadedfrom the dataset.
        """
        
        # coordinate transformation 
        if self.dataset == 'StereoMIS':
            poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
            try:
                poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
            except: 
                # No far and near
                poses = poses_arr.reshape([-1, 3, 5])  # (N_cams, 3, 5)
            # StereoMIS has well calibrated intrinsics
            cy, cx, focal =  poses[0, :, -1]
            cy = 512//2
            cx = 640//2
            focal = focal / self.downsample
            self.focal = (focal, focal)
            self.K = np.array([[focal, 0 , cx],
                                        [0, focal, cy],
                                        [0, 0, 1]]).astype(np.float32)
            poses = np.concatenate([poses[..., :1], poses[..., 1:2], poses[..., 2:3], poses[..., 3:4]], -1)
        elif self.dataset == 'EndoNeRF':
            # load poses
            poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
            poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
            H, W, focal = poses[0, :, -1]
            focal = focal / self.downsample
            self.focal = (focal, focal)
            self.K = np.array([[focal, 0 , W//2],
                                        [0, focal, H//2],
                                        [0, 0, 1]]).astype(np.float32)
            poses = np.concatenate([poses[..., :1], poses[..., 1:2], poses[..., 2:3], poses[..., 3:4]], -1)
        else:
            assert 0, NotImplemented

        if self.debug_reduce:
            poses = poses[:self.reduced_to]
        
        # prepare poses
        self.image_poses = []
        self.image_times = []
        for idx in range(poses.shape[0]):
            pose = poses[idx]
            c2w = np.concatenate((pose, np.array([[0, 0, 0, 1]])), axis=0) # 4x4
            
            # # ======================Generate the novel view for infer (StereoMIS)==========================
            # thetas = np.linspace(0, 30, poses.shape[0], endpoint=False)
            # c2w = update_extr(c2w, rotation_deg=thetas[idx], radii_mm=30)
            # # =================================================================================
            
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            T = w2c[:3, -1] #w2c
            R = np.transpose(R) #c2w
            self.image_poses.append((R, T))
            self.image_times.append(idx / poses.shape[0])
            
        
        # get paths of images, depths, masks, etc.
        agg_fn = lambda filetype: sorted(glob.glob(os.path.join(self.root_dir, filetype, "*.png")))
        self.image_paths = agg_fn("images")
        self.depth_paths = agg_fn("depth")
        self.merged_masks_paths = agg_fn("masks_merged")#all_masks
        
        if self.debug_reduce:
            self.image_paths = agg_fn("images")[:self.reduced_to]
            self.depth_paths = agg_fn("depth")[:self.reduced_to]
            self.merged_masks_paths = agg_fn("masks_merged")[:self.reduced_to]#all_masks

        self.id_of_objs = glob.glob(os.path.join(self.root_dir, "masks_obj*"))
        self.num_of_objs = len(self.id_of_objs)
        assert self.num_of_objs>=1,f'{os.path.join(self.root_dir, "masks_obj*")} {self.num_of_objs}'
        self.masks_paths_dict = {'objs_all':self.merged_masks_paths}
        for i in range(self.num_of_objs):
            self.masks_paths_dict[f'obj_tool{i+1}'] = agg_fn(f"masks_obj{i+1}") if not self.debug_reduce else agg_fn(f"masks_obj{i+1}")[:self.reduced_to]
            assert len(self.masks_paths_dict[f'obj_tool{i+1}'])==len(self.merged_masks_paths),f'{len(self.masks_paths_dict[f"obj_tool{i+1}"])} {len(self.merged_masks_paths)}'

        assert len(self.image_paths) == poses.shape[0], "the number of images should equal to the number of poses"
        assert len(self.depth_paths) == poses.shape[0], "the number of depth images should equal to number of poses"
        assert len(self.merged_masks_paths) == poses.shape[0], f"the number of masks should equal to the number of poses {len(self.merged_masks_paths)} {poses.shape[0]}"

    def get_caminfo(self, split):
        # break down
        # cameras = []
        if split == 'train': idxs = self.train_idxs
        elif split == 'test': idxs = self.test_idxs
        else:
            idxs = self.video_idxs
        
        cam_infos = []
        for idx in tqdm(idxs):
            #load obj_tool masks
            obj_masks = {}
            for i in range(self.num_of_objs):
                obj_mask_path = self.masks_paths_dict[f'obj_tool{i+1}'][idx]
                obj_mask = Image.open(obj_mask_path)
                obj_mask = np.array(obj_mask) / 255.0 if self.dataset == 'EndoNeRF' else 1-np.array(obj_mask) / 255.0
                obj_masks[f'raw_obj_tool{i+1}'] = self.transform(obj_mask).bool()
                assert len(self.image_paths) == len(self.masks_paths_dict[f'obj_tool{i+1}'])

            # mask / depth
            mask_path = self.merged_masks_paths[idx]
            mask = Image.open(mask_path)
            #mask here refer to tissue are valued
            assert self.dataset in ['StereoMIS','EndoNeRF']
            #////////////////////////////////////////
            debug_endonerf_merged_mask = True
            debug_endonerf_merged_mask = False
            if debug_endonerf_merged_mask:
                if self.dataset == 'EndoNeRF':
                    #merge masks in obj_masks
                    mask = np.zeros_like(mask).bool()
                    for i in range(self.num_of_objs):
                        mask = mask | obj_masks[f'raw_obj_tool{i+1}']
                    #conver to np array of int16
                    mask = mask.int().numpy()
                    # np.array(mask) / 255.0 
            raw_tissue = np.array(mask) / 255.0 if self.dataset == 'StereoMIS' else 1-np.array(mask) / 255.0 #np.array(mask)
            raw_tissue = self.transform(raw_tissue).bool()

            #mask refer to tool are valued
            if self.dataset in ['StereoMIS']:
                mask = 255-np.array(mask) 
            elif self.dataset in ['EndoNeRF']:
                mask = np.array(mask)  
            else:
                assert 0, NotImplementedError

            if self.tool_mask == 'use':
                mask = 1 - np.array(mask) / 255.0
            elif self.tool_mask == 'inverse':
                mask = np.array(mask) / 255.0
            elif self.tool_mask == 'nouse':
                mask = np.ones_like(mask)
            else:
                assert 0
            assert len(mask.shape)==2
        
            depth_path = self.depth_paths[idx]
            depth = np.array(Image.open(depth_path))

            close_depth,inf_depth = self.get_init_close_n_inf_depth_percentile(depth=depth,
                                                                            during_which='supervise_cam',
                                                                            )                # assert 0, 'improper'
            # ///////////////////////////////// 
            depth_sudden_change_mask = None# we use this only for futher filter tool!
            if self.supervise_depth_noise_ignore_tgt!=[]:
                depth_copy = np.copy(depth).astype(np.float32)
                depth_copy[depth_copy!=0] = 1 / depth_copy[depth_copy!=0]

                depth_sudden_change_mask = get_depth_sudden_change_mask(depth_copy,
                                                                        sobel_k_size = 1,
                                                                        grad_trd=0.001,
                                                                        # do_blur=True,
                                                                        do_blur=False,
                                                                        do_erode=False,
                                                                        do_dialte=False,
                                                                        # debug_vis_depth_sobel=True,
                                                                        debug_vis_depth_sobel=False,
                                                                        )
                depth_sudden_change_mask = self.transform(depth_sudden_change_mask).bool()
                

            
            depth = np.clip(depth, close_depth, inf_depth)# wo any ignore
            depth = torch.from_numpy(depth)
            mask = self.transform(mask).bool()
            # color
            img_path = self.image_paths[idx]
            color = np.array(Image.open(self.image_paths[idx]))/255.0
            image = self.transform(color)
            # times           
            time = self.image_times[idx]
            # poses
            R, T = self.image_poses[idx]
            # fov
            FovX = focal2fov(self.focal[0], self.img_wh[0])
            FovY = focal2fov(self.focal[1], self.img_wh[1])
            
            # mostimporantly: gen the required cam_metadata by ttgs
            pose = np.eye(4)
            pose[:3,:3] = R
            pose[:3,-1] = T
            cam_metadata = dict()
            cam_metadata['frame'] = image#frames[i]
            cam_metadata['cam'] = '0',#idx,#cams[i]
            cam_metadata['frame_idx'] = idx #frames_idx[i]
            cam_metadata['ego_pose'] = pose
            cam_metadata['extrinsic'] = self.K
            cam_metadata['timestamp'] = idx #cams_timestamps[i]
            if idx in self.train_idxs:
                cam_metadata['is_val'] = False
            else:
                cam_metadata['is_val'] = True

            from scene.dataset_readers import CameraInfo

            masks = {
                "raw_tissue":raw_tissue,
            }
            masks.update(obj_masks)


            cam_info = CameraInfo(
                    R=R, 
                    T=T, 
                    FovX=FovX, 
                    FovY=FovY, 
                    K=self.K,
                    image=image, 
                    image_name=f"{idx}", 
                    metadata=cam_metadata,
                    image_path=img_path,
                    
                    #exclusive to ttgs
                    id=idx, 
                    # acc_mask
                    #exclusive to deform3dgs
                    uid=idx,
                    time=time,
                    mask=mask, 
                    depth=depth, 
                    gt_alpha_mask=None,
                    data_device=torch.device("cuda"), 
                    Znear=None, Zfar=None, 
                    h=self.img_wh[1], w=self.img_wh[0],
                    masks = masks,
                    dataset_name_for_different_z = self.dataset,
                    depth_sudden_change_mask = depth_sudden_change_mask,
                )
            cam_infos.append(cam_info)

        return cam_infos


    def format_infos(self, split):
        # prepare the required cam_infos by ttgs
        cam_infos = self.get_caminfo(split=split)

        from utils.camera_utils import cameraList_from_camInfos
        #we rewrite the cameralist_from_caminfo func        
        return cameraList_from_camInfos(cam_infos)

    
    def filling_pts_colors(self, filling_mask, ref_depth, ref_image):
         # bool
        refined_depth = inpaint_depth(ref_depth, filling_mask)
        refined_rgb = inpaint_rgb(ref_image, filling_mask)
        return refined_rgb, refined_depth

    def get_init_close_n_inf_depth_percentile(self,depth,
                                              during_which,                                
                                              
                                              mode = 'percentile',
                                              close_depth_percent_f1 = 0.1,
                                              inf_depth_percent_f1 = 99.9,
                                              close_depth_percent_other_fusion=3.0,
                                              inf_depth_percent_other_fusion=99.8,
                                              close_depth_percent_supervise_cam=3.0,
                                              inf_depth_percent_supervise_cam=99.8,
                                            #   mode = 'Sober',
                                              ):
        assert during_which in ['f1','other_fusion','supervise_cam'],during_which
        if during_which == 'other_fusion':
            close_depth_percent = close_depth_percent_other_fusion
            inf_depth_percent = inf_depth_percent_other_fusion
    
        elif during_which == 'supervise_cam':
            # related to the do_tool_supervise_noise_removal in loss_computation!
            close_depth_percent = close_depth_percent_supervise_cam
            inf_depth_percent = inf_depth_percent_supervise_cam
    
        elif during_which == 'f1':
            close_depth_percent = close_depth_percent_f1
            inf_depth_percent = inf_depth_percent_f1

        else:
            assert 0,during_which

        assert mode in ['percentile','Sober']
        if mode == 'percentile':
            close_depth = np.percentile(depth[depth!=0], close_depth_percent)
            inf_depth = np.percentile(depth[depth!=0], inf_depth_percent)  
        elif mode == 'Sober':
            assert 0, NotImplementedError
            # Detect depth discontinuities using Sobel operator
            sobel_depth = cv2.Sobel(depth, cv2.CV_64F, 1, 1)
            sobel_depth = np.abs(sobel_depth)
            
            # Smooth the edge response
            smoothed_depth = cv2.GaussianBlur(sobel_depth, (3, 3), cv2.BORDER_DEFAULT)
            
            # Create mask for high gradient regions (potential outliers)
            # 99 Only extreme gradients (top 1%)
            depth_threshold = np.percentile(smoothed_depth[depth != 0], 98)  # Adaptive threshold
            outlier_mask = smoothed_depth > depth_threshold
            
            # Dilate the mask to include neighboring pixels
            kernel = np.ones((3, 3), np.uint8)
            # erode to remove noise
            outlier_mask = cv2.erode(outlier_mask.astype(np.uint8), kernel, iterations=3)
            # outlier_mask = cv2.dilate(outlier_mask.astype(np.uint8), kernel, iterations=1)
            
            # Get valid depth range from non-outlier regions
            valid_depths = depth[~outlier_mask & (depth != 0)]

            debug_vis = True
            if debug_vis:

                # Visualization
                import matplotlib.pyplot as plt
            
                plt.figure(figsize=(15, 5))
                
                # Original depth
                plt.subplot(131)
                plt.imshow(depth, cmap='viridis')
                plt.colorbar()
                plt.title('Original Depth')
                
                # Filtered depth (with outliers removed)
                filtered_depth = depth.copy()
                filtered_depth[outlier_mask] = 1
                plt.subplot(132)
                plt.imshow(filtered_depth, cmap='viridis')
                plt.colorbar()
                plt.title('Filtered Depth')
                
                # Removed regions (outliers)
                removed_depth = depth.copy()
                removed_depth[~outlier_mask] = 0
                plt.subplot(133)
                plt.imshow(removed_depth, cmap='viridis')
                plt.colorbar()
                plt.title('Removed Outliers')
                
                plt.tight_layout()
                plt.savefig('depth_filtering_visualization.png')
                plt.close()

            if len(valid_depths) > 0:
                close_depth = np.percentile(valid_depths, 1)
                inf_depth = np.percentile(valid_depths, 99)
            else:
                assert 0
                close_depth = 0.1
                inf_depth = 10

            
        return close_depth,inf_depth    
    
    def get_sparse_pts_dict_ttgs(self, sample=True, tissue_init_mode = None,tool_init_mode = None,
                                   inited_pcd_noise_removal = False,
                                   save_each_tool_ply = False,
                                   save_dense = False,
                                   save_ignore_depth_filter = False,
                                   each_ply_path = None,
                                  ):
        R, T = self.image_poses[0]
        depth = np.array(Image.open(self.depth_paths[0]))
        depth_mask = np.ones(depth.shape).astype(np.float32)
        close_depth,inf_depth = self.get_init_close_n_inf_depth_percentile(depth=depth,
                                                                           during_which='f1',
                                                                           )
        depth_mask[depth>inf_depth] = 0
        depth_mask[np.bitwise_and(depth<close_depth, depth!=0)] = 0
        depth_mask[depth==0] = 0
        depth[depth_mask==0] = 0

        init_masks_dict = {}

        for mask_key in self.masks_paths_dict.keys():
            #use mask in init too
            # mask = Image.open(self.masks_paths[0])
            mask = Image.open(self.masks_paths_dict[mask_key][0])
            #mask refer to tool are valued
            if self.dataset in ['StereoMIS']:
                mask = 255-np.array(mask) 
            elif self.dataset in ['EndoNeRF']:
                mask = np.array(mask)  
            else:
                assert 0, NotImplementedError
            if self.tool_mask == 'use':
                mask = 1 - np.array(mask) / 255.0
            elif self.tool_mask == 'inverse':
                mask = np.array(mask) / 255.0
            elif self.tool_mask == 'nouse':
                mask = np.ones_like(mask)
            else:
                assert 0
            assert len(mask.shape)==2

            if mask_key == 'objs_all':
                assert self.process_tissue_mask_init in [None,'erode']
                # erode mask with scipy
                if self.process_tissue_mask_init == 'erode':
                    from utils.general_utils import erode_mask_torch
                    mask = erode_mask_torch(masks = torch.Tensor(mask).bool().unsqueeze(0),\
                                            kernel_size = 120).squeeze(0).numpy().astype(mask.dtype)
                else:
                    mask = mask

                init_masks_dict['tissue']= mask
            elif mask_key.startswith('obj_tool'):
                init_masks_dict[mask_key] = 1-mask
            else:
                assert 0, mask_key

        fallback_debug = True
        fallback_debug = False
        if fallback_debug:
            init_masks_dict['obj_tool1']= init_masks_dict['objs_all']

        pts_dict = {}
        colors_dict = {}
        normals_dict = {}
        for piece_name,piece_mask in init_masks_dict.items():
            if piece_name == 'objs_all':
                print('no init for objs_all...')
                continue
            mask = np.logical_and(depth_mask, piece_mask)  
            color = np.array(Image.open(self.image_paths[0]))/255.0
            pts, colors, _ = self.get_pts_cam(depth, mask, color, disable_mask=False)
            c2w = self.get_camera_poses((R, T))
            pts = self.transform_cam2cam(pts, c2w)
            
            if piece_name == 'tissue':
                init_mode = tissue_init_mode
                assert init_mode in ['MAPF','adaptedMAPF','skipMAPF','rand']
            elif 'obj_tool' in piece_name:
                init_mode = tool_init_mode
                assert init_mode in ['MAPF','adaptedMAPF','skipMAPF','rand','TF','learnedTF']
            else:
                assert 0, piece_name

            if init_mode=='skipMAPF':
                # allowed of consider mask
                pts = pts
                colors = colors
            elif init_mode == 'learnedTF':
                print('Tool fusion init...')
                pt_dir = self.root_dir
                pt_file = 'all_obj_tools_learned.pt'
                pt_path = os.path.join(pt_dir,pt_file)
                assert os.path.exists(pt_path),f'{pt_path}'
                learned_obj_tracklets = self.load_corrected_obj_tracklets(pt_path=pt_path)
                assert learned_obj_tracklets!={}

                pts, colors = self.search_pts_colors_with_tool_mask(pts, colors, mask, c2w, 
                                                                    transform_with_tool_pose = True,
                                                                    piece_name = piece_name,
                                                                    learned_obj_tracklets = learned_obj_tracklets,
                                                                    top_k_samples=-1,
                                                                    # top_k_samples=5,
                                                                    interval=1,
                                                                    # interval=5,
                                                                    )#TF
            elif init_mode == 'TF':
                print('Tool fusion init...')
                pts, colors = self.search_pts_colors_with_tool_mask(pts, colors, mask, c2w, 
                                                                    transform_with_tool_pose = True,
                                                                    piece_name = piece_name,
                                                                    learned_obj_tracklets = None,
                                                                    top_k_samples=-1,
                                                                    interval=1,
                                                                    save_each_tool_ply = save_each_tool_ply,
                                                                    each_ply_path = each_ply_path,
                                                                    save_dense = save_dense,
                                                                    save_ignore_depth_filter = save_ignore_depth_filter,
                                                                    # top_k_samples=5,                                                            
                                                                    # piece_name = 'objs_all',
                                                                    )#TF
            elif init_mode == 'MAPF':
                # allowed of consider mask
                pts, colors = self.search_pts_colors_with_motion(pts, colors, mask, c2w)#MAPF
            elif init_mode == 'adaptedMAPF':
                # allowed of consider mask
                occlu_interval = self.init_detail_params_dict['occlu_interval']
                deform_interval = self.init_detail_params_dict['deform_interval']
                add4occlu = self.init_detail_params_dict['add4occlu']
                add4deform = self.init_detail_params_dict['add4deform']
                pts, colors = self.search_pts_colors_with_motion_adapted(pts, colors, mask, c2w, 
                                                                        occlu_interval=occlu_interval,
                                                                        deform_interval=deform_interval,
                                                                        add4occlu = add4occlu,
                                                                        add4deform = add4deform
                                                                         )#MAPF
            elif init_mode == 'online_missingonly_OAPF':
                # allowed of consider mask
                pts, colors = self.search_pts_colors_with_motion_missingonly_online(pts, colors, mask, c2w, 
                                                                         occlu_interval=8, 
                                                                         add4deform = True
                                                                         )#MAPF                                                                        
            elif init_mode == 'rand':
                #/////////////////////////////////////////
                rand_num_pts = 100_000  
                rand_num_pts = 10_000  
                warnings.warn(f"tissue rand init(w.o concerning mask): generating random point cloud ({rand_num_pts})... w.o mask constrains?")
                # use the params from deformable-3d-gs synthetic Blender scenes
                pts = np.random.random((rand_num_pts, 3)) * 2.6 - 1.3
                shs = np.random.random((rand_num_pts, 3)) / 255.0
                colors=SH2RGB(shs)
                #//////////////////////
            else:
                assert 0, NotImplementedError
            normals = np.zeros((pts.shape[0], 3))

            if sample:
                num_sample = int(0.1 * pts.shape[0])
                sel_idxs = np.random.choice(pts.shape[0], num_sample, replace=False)
                pts = pts[sel_idxs, :]
                colors = colors[sel_idxs, :]
                normals = normals[sel_idxs, :]
            #/////////////////////
            if inited_pcd_noise_removal:
                assert len(pts)==len(colors)==len(normals)
                valid_mask = self.init_points_noise_removal(pts,
                                                            # dbg_vis_init_noise_removal = True,
                                                            piece_name_in_title= piece_name)
                pts = pts[valid_mask]
                colors = colors[valid_mask]
                normals = normals[valid_mask]

            pts_dict[piece_name] = pts
            colors_dict[piece_name] = colors
            normals_dict[piece_name] = normals
        # return pts, colors, normals
        # return pts_dict,colors_dict,normals_dict
        return pts_dict,colors_dict,normals_dict, init_masks_dict


    def get_sparse_pts(self, sample=True, tissue_init_mode = None,tool_init_mode = None,
                       inited_pcd_noise_removal = False,
                       ):
        init_mode = tissue_init_mode

        assert init_mode in ['MAPF','adaptedMAPF','skipMAPF','rand','TF']
        R, T = self.image_poses[0]
        depth = np.array(Image.open(self.depth_paths[0]))
        depth_mask = np.ones(depth.shape).astype(np.float32)
        close_depth,inf_depth = self.get_init_close_n_inf_depth_percentile(depth=depth,
                                                                           during_which='f1',)


        depth_mask[depth>inf_depth] = 0
        depth_mask[np.bitwise_and(depth<close_depth, depth!=0)] = 0
        depth_mask[depth==0] = 0
        depth[depth_mask==0] = 0

        mask = Image.open(self.merged_masks_paths[0])
        if self.dataset in ['StereoMIS']:
            mask = 255-np.array(mask) 
        elif self.dataset in ['EndoNeRF']:
            mask = np.array(mask)  
        else:
            assert 0, NotImplementedError

        if self.tool_mask == 'use':
            mask = 1 - np.array(mask) / 255.0
        elif self.tool_mask == 'inverse':
            mask = np.array(mask) / 255.0
        elif self.tool_mask == 'nouse':
            mask = np.ones_like(mask)
        else:
            assert 0
        assert len(mask.shape)==2

        assert self.process_tissue_mask_init in [None,'erode']
        if self.process_tissue_mask_init == 'erode':
            from utils.general_utils import erode_mask_torch
            mask = erode_mask_torch(masks = torch.Tensor(mask).bool().unsqueeze(0),
            kernel_size = 50).squeeze(0).numpy().astype(mask.dtype)
        else:
            mask = mask

        mask = np.logical_and(depth_mask, mask)   
        color = np.array(Image.open(self.image_paths[0]))/255.0
        pts, colors, _ = self.get_pts_cam(depth, mask, color, disable_mask=False)
        c2w = self.get_camera_poses((R, T))
        pts = self.transform_cam2cam(pts, c2w)
        
        if init_mode=='skipMAPF':
            pass
        elif init_mode == 'MAPF':
            # allowed of consider mask
            pts, colors = self.search_pts_colors_with_motion(pts, colors, mask, c2w)#MAPF
        elif init_mode == 'adaptedMAPF':
            occlu_interval = self.init_detail_params_dict['occlu_interval']
            deform_interval = self.init_detail_params_dict['deform_interval']
            add4occlu = self.init_detail_params_dict['add4occlu']
            add4deform = self.init_detail_params_dict['add4deform']
            pts, colors = self.search_pts_colors_with_motion_adapted(pts, colors, mask, c2w, 
                                                                    occlu_interval=occlu_interval,
                                                                    deform_interval=deform_interval,
                                                                    add4occlu = add4occlu,
                                                                    add4deform = add4deform
                                                                        )#MAPF
        elif init_mode == 'rand':
            rand_num_pts = 100_000
            warnings.warn(f"tissue rand init(w.o concerning mask): generating random point cloud ({rand_num_pts})... w.o mask constrains?")
            pts = np.random.random((rand_num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((rand_num_pts, 3)) / 255.0
            colors=SH2RGB(shs)
        else:
            assert 0, NotImplementedError

        normals = np.zeros((pts.shape[0], 3))

        if sample:
            num_sample = int(0.1 * pts.shape[0])
            sel_idxs = np.random.choice(pts.shape[0], num_sample, replace=False)
            pts = pts[sel_idxs, :]
            colors = colors[sel_idxs, :]
            normals = normals[sel_idxs, :]

        #/////////////////////
        if inited_pcd_noise_removal:
            assert len(pts)==len(colors)==len(normals)
            valid_mask = self.init_points_noise_removal(pts,
                                                        # dbg_vis_init_noise_removal = False,
                                                        piece_name_in_title = 'deform3dgs-tissue')
            
            pts = pts[valid_mask]
            colors = colors[valid_mask]
            normals = normals[valid_mask]


        return pts, colors, normals
    
    def init_points_noise_removal(self,pts,
                                dbg_vis_init_noise_removal = False,
                                piece_name_in_title = '',
                                mode = 'BB'
                                ):
        assert mode in ['cluster','BB']
        from utils.scene_utils import clustering_and_get_BBox_via_PCA
        center, R, min_bounds, max_bounds, valid_mask,filtered_points = clustering_and_get_BBox_via_PCA(pts,min_samples=1,
                                                                                            margin_scale=0.0,
                                                                                            cluster_eps = 0.5,
                                                                                            )
        

        if dbg_vis_init_noise_removal:
            vertices = get_vertices_from_min_max_bounds(min_bounds = min_bounds, max_bounds = max_bounds,
                                                        conduct_local2world=True,
                                                        R=R,
                                                        center=center,)

            plot_6d_bbox_with_pts(vertices=vertices,
                                    points_inside=valid_mask,
                                    points=pts,
                                    #for naming
                                vis_always_stay = True,
                                tool_adc_mode = f'NOISE REMOVAL INIT{piece_name_in_title} {len(filtered_points)}/{len(pts)}',
                                    )          
        return valid_mask      
                
                
        

    def calculate_motion_masks(self, ):
        images = []
        for j in range(0, len(self.image_poses)):
            color = np.array(Image.open(self.image_paths[j]))/255.0
            images.append(color)
        images = np.asarray(images).mean(axis=-1)
        diff_map = np.abs(images - images.mean(axis=0))
        diff_thrshold = np.percentile(diff_map[diff_map!=0], 95)
        return diff_map > diff_thrshold
        
    def search_pts_colors_with_tool_mask(self, ref_pts, ref_color, ref_mask, ref_c2w,
                                         transform_with_tool_pose = True,
                                         piece_name = None,
                                         learned_obj_tracklets = None,
                                         interval = 1,
                                         top_k_samples = -1,
                                         save_each_tool_ply = False,
                                         each_ply_path = None,
                                         save_dense = False,
                                         save_ignore_depth_filter = False,

                                         ):
        assert top_k_samples == -1 or top_k_samples>=1
        masks_paths = self.masks_paths_dict[piece_name] 
        assert transform_with_tool_pose == True
        if transform_with_tool_pose:
            assert self.obj_tracklets != None


        if learned_obj_tracklets != None:
            self.cotrackerpnp_trajectory_cams2w = learned_obj_tracklets[f'{piece_name}']['trajectory_cams2w'].float().cuda()# 
            assert self.cotrackerpnp_trajectory_cams2w.shape == self.obj_tracklets[f'{piece_name}']['trajectory_cams2w'].shape
        else:
            self.cotrackerpnp_trajectory_cams2w = self.obj_tracklets[f'{piece_name}']['trajectory_cams2w'].float().cuda()# 
        load_num,_,_ = self.cotrackerpnp_trajectory_cams2w.shape
        assert load_num == len(self.image_poses)
        cotrackerpnp_trajectory_w2cams2 = torch.linalg.inv(self.cotrackerpnp_trajectory_cams2w)

        total_cand_nums = len(self.image_poses)//interval
        percent_to_keep_from_each_candidate = 1./float(total_cand_nums)

        for j in range(1,  len(self.image_poses), interval):
            if top_k_samples!=-1 and j>top_k_samples:
                break

            R, T = self.image_poses[j]
            c2w = self.get_camera_poses((R, T))
            c2ref = np.linalg.inv(ref_c2w) @ c2w# the part comes from cam ego motion
            # extend 
            obj_6DoF = self.cotrackerpnp_trajectory_cams2w[j].cpu().numpy()
            # cotrackerpnp_trajectory_w2cams2
            c2ref = obj_6DoF @ c2ref# the part comes from self motion
            depth = np.array(Image.open(self.depth_paths[j]))
            color = np.array(Image.open(self.image_paths[j]))/255.0

            mask = Image.open(masks_paths[j])
            #resulted mask refer to tool are valued
            if self.dataset in ['StereoMIS']:
                mask = 255-np.array(mask) 
            elif self.dataset in ['EndoNeRF']:
                mask = np.array(mask)  
            else:
                assert 0, NotImplementedError

            if self.tool_mask == 'use':
                mask = 1 - np.array(mask) / 255.0
            elif self.tool_mask == 'inverse':
                mask = np.array(mask) / 255.0
            elif self.tool_mask == 'nouse':
                mask = np.ones_like(mask)
            else:
                assert 0

            # extend            
            ref_mask_not = (1-mask).astype(bool) 
            mask = np.ones_like(mask)

            assert len(mask.shape)==2
            depth_mask = np.ones(depth.shape).astype(np.float32)

            close_depth,inf_depth = self.get_init_close_n_inf_depth_percentile(depth=depth,
                                                                               during_which='other_fusion',
                                                                            )
                                                                               
            depth_mask[depth>inf_depth] = 0
            depth_mask[np.bitwise_and(depth<close_depth, depth!=0)] = 0
            depth_mask[depth==0] = 0

            mask = np.logical_and(depth_mask, mask)
            depth[mask==0] = 0
            
            pts, colors, mask_refine = self.get_pts_cam(depth, mask, color)
            pts = self.transform_cam2cam(pts, c2ref) # Nx3
            X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]
            X, Y, Z = X[Z!=0], Y[Z!=0], Z[Z!=0]
            X_Z, Y_Z = X / Z, Y / Z
            X_Z = (X_Z * self.focal[0] + self.K[0,-1]).astype(np.int32)
            Y_Z = (Y_Z * self.focal[1] + self.K[1,-1]).astype(np.int32)
            # Out of the visibility
            out_vis_mask = ((X_Z > (self.img_wh[0]-1)) + (X_Z < 0) +\
                    (Y_Z > (self.img_wh[1]-1)) + (Y_Z < 0))>0
            out_vis_pt_idx = np.where(out_vis_mask)[0]
            visible_mask = (1 - out_vis_mask)>0
            X_Z = np.clip(X_Z, 0, self.img_wh[0]-1)
            Y_Z = np.clip(Y_Z, 0, self.img_wh[1]-1)
            coords = np.stack((Y_Z, X_Z), axis=-1)
            proj_mask = np.zeros((self.img_wh[1], self.img_wh[0])).astype(np.float32)
            proj_mask[coords[:, 0], coords[:, 1]] = 1
            compl_mask = (ref_mask_not * proj_mask)
            index_mask = compl_mask.reshape(-1)[mask_refine]
            compl_idxs = np.nonzero(index_mask.reshape(-1))[0]
            if compl_idxs.shape[0] <= 50:
                continue
            compl_pts = pts[compl_idxs, :]
            compl_pts = self.transform_cam2cam(compl_pts, ref_c2w)
            compl_colors = colors[compl_idxs, :]
            sel_idxs = np.random.choice(compl_pts.shape[0], int(percent_to_keep_from_each_candidate*compl_pts.shape[0]), replace=True)
            if save_each_tool_ply:
                from scene.flexible_deform_model import BasicPointCloud
                from scene.dataset_readers import storePly
                if save_dense:
                    dense_percent = 0.5
                    save_sel_idxs = np.random.choice(compl_pts.shape[0], int(dense_percent*compl_pts.shape[0]), replace=True)
                else:
                    save_sel_idxs = sel_idxs

                xyz_i = compl_pts[save_sel_idxs]
                rgb_i = compl_colors[save_sel_idxs]
                normals_i = np.random.random((xyz_i.shape[0], 3))

                ply_path_after = os.path.join(each_ply_path, f'{piece_name}_{j}th_pose_After.ply')
                pcd_i = BasicPointCloud(points=xyz_i, colors=rgb_i, normals=normals_i)
                plydata_i = storePly(ply_path_after, xyz_i,rgb_i*255, wo_write=False)  # the points3d.ply is not used at all, try not touch the src dataset
                print('ply saved in', ply_path_after)

                ply_path_before = os.path.join(each_ply_path, f'{piece_name}_{j}th_pose_Before.ply')
                xyz_i_before = self.transform_cam2cam(xyz_i, np.linalg.inv(c2ref)) # Nx3
                rgb_i_before = rgb_i
                normals_i_before = normals_i
                plydata_i = storePly(ply_path_before, xyz_i_before,rgb_i_before*255, wo_write=False)

            ref_pts = np.concatenate((ref_pts, compl_pts[sel_idxs]), axis=0)
            ref_color = np.concatenate((ref_color, compl_colors[sel_idxs]), axis=0)
            ref_mask = np.logical_or(ref_mask, compl_mask)
        
        if ref_pts.shape[0] > 40000:
            sel_idxs = np.random.choice(ref_pts.shape[0], 30000, replace=True)  
            ref_pts = ref_pts[sel_idxs]         
            ref_color = ref_color[sel_idxs] 
        return ref_pts, ref_color
    
    def search_pts_colors_with_motion_adapted(self, ref_pts, ref_color, ref_mask, ref_c2w, 
                                               occlu_interval = 8,
                                               deform_interval = 1,
                                               add4deform = False,
                                               add4occlu = True,
                                               filter_outlier = False,
                                               ):

        ref_mask_not_ori = np.logical_not(ref_mask)#.copy()
        if add4deform:
            motion_masks = self.calculate_motion_masks()# the original implementation also consider the tool motion difference there

        added_ref_pts_occluded_list = []
        added_ref_color_occluded_list = []

        added_ref_pts_highdeformed_list = []
        added_ref_color_highdeformed_list = []

        for j in range(1,  len(self.image_poses)):
            try_add_occlu_flag = (j % occlu_interval == 0) and add4occlu
            try_add_deform_flag = (j % deform_interval == 0) and add4deform
            if not try_add_occlu_flag and not try_add_deform_flag:
                continue

            ref_mask_not = np.logical_not(ref_mask)
            ref_mask_not = np.logical_or(ref_mask_not, ref_mask_not_ori)
            R, T = self.image_poses[j]
            c2w = self.get_camera_poses((R, T))
            c2ref = np.linalg.inv(ref_c2w) @ c2w
            depth = np.array(Image.open(self.depth_paths[j]))
            color = np.array(Image.open(self.image_paths[j]))/255.0
            mask = Image.open(self.merged_masks_paths[j])            
            if self.dataset in ['StereoMIS']:
                mask = 255-np.array(mask) 
            elif self.dataset in ['EndoNeRF']:
                mask = np.array(mask)  
            else:
                assert 0, NotImplementedError
                
            if self.tool_mask == 'use':
                mask = 1 - np.array(mask) / 255.0
            elif self.tool_mask == 'inverse':
                mask = np.array(mask) / 255.0
            elif self.tool_mask == 'nouse':
                mask = np.ones_like(mask)
            else:
                assert 0
            assert len(mask.shape)==2
                        
            depth_mask = np.ones(depth.shape).astype(np.float32)
            
            close_depth,inf_depth = self.get_init_close_n_inf_depth_percentile(depth=depth,
                                                                           during_which='other_fusion',
                                                                           )

            depth_mask[depth>inf_depth] = 0
            depth_mask[np.bitwise_and(depth<close_depth, depth!=0)] = 0
            depth_mask[depth==0] = 0
            mask = np.logical_and(depth_mask, mask)
            depth[mask==0] = 0
            
            pts, colors, mask_refine = self.get_pts_cam(depth, mask, color)
            pts = self.transform_cam2cam(pts, c2ref) # Nx3
            X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]
            X, Y, Z = X[Z!=0], Y[Z!=0], Z[Z!=0]
            X_Z, Y_Z = X / Z, Y / Z
            X_Z = (X_Z * self.focal[0] + self.K[0,-1]).astype(np.int32)
            Y_Z = (Y_Z * self.focal[1] + self.K[1,-1]).astype(np.int32)
            # Out of the visibility
            out_vis_mask = ((X_Z > (self.img_wh[0]-1)) + (X_Z < 0) +\
                    (Y_Z > (self.img_wh[1]-1)) + (Y_Z < 0))>0
            out_vis_pt_idx = np.where(out_vis_mask)[0]
            visible_mask = (1 - out_vis_mask)>0
            X_Z = np.clip(X_Z, 0, self.img_wh[0]-1)
            Y_Z = np.clip(Y_Z, 0, self.img_wh[1]-1)
            coords = np.stack((Y_Z, X_Z), axis=-1)
            proj_mask = np.zeros((self.img_wh[1], self.img_wh[0])).astype(np.float32)
            proj_mask[coords[:, 0], coords[:, 1]] = 1

            #/////////////////////////////////
            if try_add_occlu_flag:
                compl_mask = (ref_mask_not * proj_mask)#historically less infoed area(have infor in current frame)
                index_mask = compl_mask.reshape(-1)[mask_refine]
                compl_idxs = np.nonzero(index_mask.reshape(-1))[0]
                if compl_idxs.shape[0] <= 50:
                    continue
                compl_pts = pts[compl_idxs, :]
                compl_pts = self.transform_cam2cam(compl_pts, ref_c2w)
                compl_colors = colors[compl_idxs, :]
                ref_mask = np.logical_or(ref_mask, compl_mask)

                percent = 1
                occlueded_sel_idxs = np.random.choice(compl_pts.shape[0], int(percent*compl_pts.shape[0]), replace=True)
                added_ref_pts_occluded_list.append(compl_pts[occlueded_sel_idxs])
                added_ref_color_occluded_list.append(compl_colors[occlueded_sel_idxs])


            if try_add_deform_flag:
                motion_mask = motion_masks[j]
                motion_mask =  np.logical_and(motion_mask,mask)
                
                deform_compl_mask = (motion_mask * proj_mask)
                deform_index_mask = deform_compl_mask.reshape(-1)[mask_refine]
                deform_compl_idxs = np.nonzero(deform_index_mask.reshape(-1))[0]
                deform_compl_pts = pts[deform_compl_idxs, :]
                deform_compl_pts = self.transform_cam2cam(deform_compl_pts, ref_c2w)
                deform_compl_colors = colors[deform_compl_idxs, :]

                added_ref_pts_highdeformed_list.append(deform_compl_pts)
                added_ref_color_highdeformed_list.append(deform_compl_colors)


        if added_ref_pts_occluded_list!=[]:
            added_ref_pts = np.concatenate(added_ref_pts_occluded_list, axis=0)
            added_ref_color = np.concatenate(added_ref_color_occluded_list, axis=0)

            if added_ref_pts.shape[0] > 1000:
                target_density = 0.2
                sel_idxs = np.random.choice(added_ref_pts.shape[0], int(added_ref_pts.shape[0]*target_density), replace=True)  
                added_ref_pts = added_ref_pts[sel_idxs]         
                added_ref_color = added_ref_color[sel_idxs] 

            ref_pts = np.concatenate((ref_pts, added_ref_pts), axis=0)
            ref_color = np.concatenate((ref_color, added_ref_color), axis=0)
        if added_ref_pts_highdeformed_list!=[]:
            deform_added_ref_pts = np.concatenate(added_ref_pts_highdeformed_list, axis=0)
            deform_added_ref_color = np.concatenate(added_ref_color_highdeformed_list, axis=0)
            ref_pts = np.concatenate((ref_pts,  deform_added_ref_pts), axis=0) #if deform_added_ref_pts!=[] else np.concatenate((ref_pts, added_ref_pts), axis=0)
            ref_color = np.concatenate((ref_color, deform_added_ref_color), axis=0) #if deform_added_ref_color!=[] else np.concatenate((ref_color, added_ref_color), axis=0)

        if ref_pts.shape[0] > 600000:
            sel_idxs = np.random.choice(ref_pts.shape[0], 500000, replace=True)  
            ref_pts = ref_pts[sel_idxs]         
            ref_color = ref_color[sel_idxs] 

        return ref_pts, ref_color
    
    def search_pts_colors_with_motion(self, ref_pts, ref_color, ref_mask, ref_c2w):
        # calculating the motion mask
        motion_mask = self.calculate_motion_masks()
        interval = 1

        if len(self.image_poses) > 150: # in case long sequence
            interval = 2
            assert 0, f"long sequence: {len(self.image_poses)}...set interval to 2?"

        for j in range(1,  len(self.image_poses), interval):
            ref_mask_not = np.logical_not(ref_mask)
            ref_mask_not = np.logical_or(ref_mask_not, motion_mask[0])
            R, T = self.image_poses[j]
            c2w = self.get_camera_poses((R, T))
            c2ref = np.linalg.inv(ref_c2w) @ c2w
            depth = np.array(Image.open(self.depth_paths[j]))
            color = np.array(Image.open(self.image_paths[j]))/255.0
            mask = Image.open(self.merged_masks_paths[j])
            if self.dataset in ['StereoMIS']:
                mask = 255-np.array(mask) 
            elif self.dataset in ['EndoNeRF']:
                mask = np.array(mask)  
            else:
                assert 0, NotImplementedError
                
            if self.tool_mask == 'use':
                mask = 1 - np.array(mask) / 255.0
            elif self.tool_mask == 'inverse':
                mask = np.array(mask) / 255.0
            elif self.tool_mask == 'nouse':
                mask = np.ones_like(mask)
            else:
                assert 0
            assert len(mask.shape)==2
            depth_mask = np.ones(depth.shape).astype(np.float32)

            close_depth,inf_depth = self.get_init_close_n_inf_depth_percentile(depth=depth,
                                                                           during_which='other_fusion',
                                                                           )

            depth_mask[depth>inf_depth] = 0
            depth_mask[np.bitwise_and(depth<close_depth, depth!=0)] = 0
            depth_mask[depth==0] = 0
            mask = np.logical_and(depth_mask, mask)
            depth[mask==0] = 0
            
            pts, colors, mask_refine = self.get_pts_cam(depth, mask, color)
            pts = self.transform_cam2cam(pts, c2ref) # Nx3
            X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]
            X, Y, Z = X[Z!=0], Y[Z!=0], Z[Z!=0]
            X_Z, Y_Z = X / Z, Y / Z
            X_Z = (X_Z * self.focal[0] + self.K[0,-1]).astype(np.int32)
            Y_Z = (Y_Z * self.focal[1] + self.K[1,-1]).astype(np.int32)
            # Out of the visibility
            out_vis_mask = ((X_Z > (self.img_wh[0]-1)) + (X_Z < 0) +\
                    (Y_Z > (self.img_wh[1]-1)) + (Y_Z < 0))>0
            out_vis_pt_idx = np.where(out_vis_mask)[0]
            visible_mask = (1 - out_vis_mask)>0
            X_Z = np.clip(X_Z, 0, self.img_wh[0]-1)
            Y_Z = np.clip(Y_Z, 0, self.img_wh[1]-1)
            coords = np.stack((Y_Z, X_Z), axis=-1)
            proj_mask = np.zeros((self.img_wh[1], self.img_wh[0])).astype(np.float32)
            proj_mask[coords[:, 0], coords[:, 1]] = 1
            compl_mask = (ref_mask_not * proj_mask)
            index_mask = compl_mask.reshape(-1)[mask_refine]
            compl_idxs = np.nonzero(index_mask.reshape(-1))[0]
            if compl_idxs.shape[0] <= 50:
                continue
            compl_pts = pts[compl_idxs, :]
            compl_pts = self.transform_cam2cam(compl_pts, ref_c2w)
            compl_colors = colors[compl_idxs, :]
            sel_idxs = np.random.choice(compl_pts.shape[0], int(0.1*compl_pts.shape[0]), replace=True)
            ref_pts = np.concatenate((ref_pts, compl_pts[sel_idxs]), axis=0)
            ref_color = np.concatenate((ref_color, compl_colors[sel_idxs]), axis=0)
            ref_mask = np.logical_or(ref_mask, compl_mask)
        
        if ref_pts.shape[0] > 600000:
            sel_idxs = np.random.choice(ref_pts.shape[0], 500000, replace=True)  
            ref_pts = ref_pts[sel_idxs]         
            ref_color = ref_color[sel_idxs] 
        return ref_pts, ref_color
    
         
    def get_camera_poses(self, pose_tuple):
        R, T = pose_tuple
        R = np.transpose(R)
        w2c = np.concatenate((R, T[...,None]), axis=-1)
        w2c = np.concatenate((w2c, np.array([[0, 0, 0, 1]])), axis=0)
        c2w = np.linalg.inv(w2c)
        return c2w
    
    def get_pts_cam(self, depth, mask, color, disable_mask=False):
        W, H = self.img_wh
        i, j = np.meshgrid(np.linspace(0, W-1, W), np.linspace(0, H-1, H))
        cx = self.K[0,-1]
        cy = self.K[1,-1]
        X_Z = (i-cx) / self.focal[0]
        Y_Z = (j-cy) / self.focal[1]
        # assert 0,f'{self.K} {self.focal}'
        Z = depth
        X, Y = X_Z * Z, Y_Z * Z
        pts_cam = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        color = color.reshape(-1, 3)
        
        if not disable_mask:
            mask = mask.reshape(-1).astype(bool)
            pts_valid = pts_cam[mask, :]
            color_valid = color[mask, :]
        else:
            mask = mask.reshape(-1).astype(bool)
            pts_valid = pts_cam
            color_valid = color
            color_valid[mask==0, :] = np.ones(3)
                    
        return pts_valid, color_valid, mask
        
    def get_maxtime(self):
        return self.maxtime
    
    def transform_cam2cam(self, pts_cam, pose):
        pts_cam_homo = np.concatenate((pts_cam, np.ones((pts_cam.shape[0], 1))), axis=-1)
        pts_wld = np.transpose(pose @ np.transpose(pts_cam_homo))
        xyz = pts_wld[:, :3]
        return xyz
    def load_other_obj_meta(self,cameras = [0],num_frames = None):
        scene_metadata = {}
        obj_tracklets = None
        tracklet_timestamps = None
        obj_info = None
        exts = []
        # #/////////////////////////////
        if self.load_cotrackerPnpPose:
            self.obj_poses_path_dict = {}
            pose_pattern = f'ObjPoses_rel_CoTracer_query_queryGenMaskframe-*_ptsN1000_PnP_LMrf0_masks_obj'
            # pose_pattern = f'ObjPoses_rel_CoTracer_query_bi_queryGenMaskframe-*_ptsN1000_PnP_LMrf1_masks_obj'
            pose_pattern = f'ObjPoses_rel_CoTracer_query_*queryGenMaskframe-*_ptsN1000_PnP_LMrf*_masks*'
            agg_fn = lambda filetype: sorted(glob.glob(os.path.join(self.root_dir, f"*{pose_pattern}*.pt")))
            cotrackerPnpPose_paths = agg_fn("images")
            assert len(cotrackerPnpPose_paths)>0,pose_pattern
            for cotrackerPnpPose_path in cotrackerPnpPose_paths:
                if 'masks_obj' in cotrackerPnpPose_path:
                    mask_dir_id = cotrackerPnpPose_path.split('masks_obj')[-1].split('.')[0]
                    tool_obj_name = F'obj_tool{mask_dir_id}'
                    assert tool_obj_name not in self.obj_poses_path_dict.keys(),'only a single file'
                    self.obj_poses_path_dict[tool_obj_name] = cotrackerPnpPose_path
                elif 'masks.pt' in cotrackerPnpPose_path:
                    tool_obj_name = F'objs_all'
                    assert tool_obj_name not in self.obj_poses_path_dict.keys(),'only a single file'
                    self.obj_poses_path_dict[tool_obj_name] = cotrackerPnpPose_path
                else:
                    assert 0, cotrackerPnpPose_path
                
            obj_tracklets = {}
            for tool_obj_name in self.obj_poses_path_dict.keys():
                if tool_obj_name == 'objs_all':
                    continue
                assert tool_obj_name.startswith('obj_tool') or tool_obj_name=='objs_all'
                obj_tracklets[tool_obj_name] = torch.load(self.obj_poses_path_dict[tool_obj_name])
                if self.debug_reduce:
                    for key,value in obj_tracklets[tool_obj_name].items():
                        obj_tracklets[tool_obj_name][key] = value[:self.reduced_to]


            self.obj_tracklets = obj_tracklets
        else:
            assert 0, 'always load even for deform3dgs--we want to do obj-wise metric logggin'

        scene_metadata = dict()
        scene_metadata['obj_tracklets'] = obj_tracklets
        scene_metadata['tracklet_timestamps'] = tracklet_timestamps
        scene_metadata['obj_meta'] = obj_info
        scene_metadata['num_images'] = len(exts)
        scene_metadata['num_cams'] = len(cameras)
        scene_metadata['num_frames'] = num_frames

        scene_metadata['camera_timestamps'] = self.camera_timestamps
        scene_metadata['all_tool_objs_name'] = [i for i in list(self.obj_poses_path_dict.keys())  if i != 'objs_all' ]#self.camera_timestamps
        assert 'objs_all' in list(self.obj_poses_path_dict.keys()),'objs_all should be included'
        scene_metadata['all_tool_objs_name_w_merged'] = [i for i in list(self.obj_poses_path_dict.keys())]#self.camera_timestamps

        return scene_metadata

    def get_endonerf_cam_meta(self,cameras = [0]):
        metadata = {}#cam_metadata
        return metadata

