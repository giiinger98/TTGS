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
import sys
from PIL import Image
from typing import NamedTuple
import torchvision.transforms as transforms
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import torch
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from scene.flexible_deform_model import BasicPointCloud
from utils.general_utils import PILtoTorch
from tqdm import tqdm

#only used for ttgs
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    K: np.array
    image: np.array
    image_path: str
    image_name: str

    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    # width: int
    # height: int
    time : float

    h: int
    w: int
    id: int
    depth: np.array
    gt_alpha_mask: np.array
    data_device : str
    Znear : np.array
    Zfar: np.array

    #ttgs related
    metadata: dict = dict()
    masks: dict = dict()
    # metadata: dict = dict()
    mask: np.array = None
    acc_mask: np.array = None
    dataset_name_for_different_z: str = ''
    depth_sudden_change_mask: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int
    
    #ttgs related
    scene_metadata: dict = dict()
    cam_metadata: dict = dict()
    point_cloud_dict: dict = dict()
    loaded_obj_names: list = []

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center

    return {"translate": translate, "radius": radius}



def fetchPly(path = None, plydata = None):
    if path == None:
        assert plydata != None
    else:
        plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb, wo_write = False):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    if not wo_write:
        ply_data.write(path)
    return ply_data

def generateCamerasFromTransforms(path, template_transformsfile, extension, maxtime):
    trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

    rot_phi = lambda phi : torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).float()

    rot_theta = lambda th : torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).float()

    def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w

    cam_infos = []
    # generate render poses and times
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    render_times = torch.linspace(0,maxtime,render_poses.shape[0])
    with open(os.path.join(path, template_transformsfile)) as json_file:
        template_json = json.load(json_file)
        fovx = template_json["camera_angle_x"]

    # load a single image to get image info.
    for idx, frame in enumerate(template_json["frames"]):
        cam_name = os.path.join(path, frame["file_path"] + extension)
        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        image = PILtoTorch(image,(800,800))
        break

    # format information
    for idx, (time, poses) in enumerate(zip(render_times,render_poses)):
        time = time/maxtime
        matrix = np.linalg.inv(np.array(poses))
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
        FovY = fovy 
        FovX = fovx
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=None, image_name=None, width=image.shape[1], height=image.shape[2],
                            time = time))
    return cam_infos

def readEndoNeRFInfo(datadir,tool_mask = 'use',
                    #  init_mode = None,
                     tissue_init_mode = None,
                     tool_init_mode = None,
                     load_other_obj_meta = False,
                     cfg = None,
                     load_pcd_dict_in_sceneinfo = False,
                     load_cotrackerPnpPose = False,
                     exp_path = None,
                     init_detail_params_dict = {},
                     process_tissue_mask_init = None,
                     inited_pcd_noise_removal = False,
                     supervise_depth_noise_ignore_tgt = [],

                     ):
    assert isinstance(init_detail_params_dict, dict)
    assert tissue_init_mode in ['adaptedMAPF','MAPF','skipMAPF','rand','TF']
    assert tool_init_mode in ['adaptedMAPF','MAPF','skipMAPF','rand','TF','learnedTF']
    from scene.endo_loader import EndoNeRF_Dataset

    # Load the cam_meta + scene_meta + poses_extrinsics
    endo_dataset = EndoNeRF_Dataset(
        datadir=datadir,
        downsample=1.0,
        tool_mask=tool_mask,
        load_cotrackerPnpPose = load_cotrackerPnpPose,
        init_detail_params_dict = init_detail_params_dict,
        process_tissue_mask_init = process_tissue_mask_init,
        model_path=exp_path,
        supervise_depth_noise_ignore_tgt = supervise_depth_noise_ignore_tgt,
    )

    print('Loading camera....(surg-gs have seperate cam and scene)')
    train_cam_infos = endo_dataset.format_infos(split="train")
    test_cam_infos = endo_dataset.format_infos(split="test")
    video_cam_infos = endo_dataset.format_infos(split="video")
    # get cam normalizations
    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_metadata = {}
    cam_metadata = {}

    scene_metadata = endo_dataset.load_other_obj_meta(cameras=[0],
                                                        num_frames=None)
    if load_other_obj_meta:
        assert cfg != None
        #udpate
        scene_metadata['scene_center'] = nerf_normalization['translate']
        scene_metadata['scene_radius'] = nerf_normalization['radius']
        sphere_normalization = nerf_normalization
        scene_metadata['sphere_center'] = sphere_normalization['translate']
        scene_metadata['sphere_radius'] = sphere_normalization['radius']

    pcd = None
    pcd_dict = {}
    if not load_pcd_dict_in_sceneinfo:
        xyz, rgb, normals = endo_dataset.get_sparse_pts(tissue_init_mode=tissue_init_mode,
                                                        tool_init_mode=tool_init_mode,
                                                        inited_pcd_noise_removal = inited_pcd_noise_removal,)

        normals = np.random.random((xyz.shape[0], 3))
        pcd = BasicPointCloud(points=xyz, colors=rgb, normals=normals)

        ply_path = os.path.join(exp_path, f"deform3dgs_points3d.ply")
        plydata = storePly(ply_path, xyz,rgb*255, wo_write=False)  # the points3d.ply is not used at all, try not touch the src dataset
        print('the points3d.ply is not used at all, try not touch the src dataset')
        try:
            pcd = fetchPly(path = None, plydata = plydata)
        except:
            pcd = None
    else:
        xyz_dict, rgb_dict, normals_dict, init_mask_dict = endo_dataset.get_sparse_pts_dict_ttgs(tissue_init_mode=tissue_init_mode,
                                                                                                  tool_init_mode=tool_init_mode,
                                                                                                  inited_pcd_noise_removal = inited_pcd_noise_removal,
                                                                                                  )
        import copy
        scene_metadata['init_mask_dict'] = copy.deepcopy(init_mask_dict)# used for tool densify condition

        xyz_fused = np.concatenate([xyz_dict[piece_name] for piece_name in xyz_dict.keys()], axis=0)
        rgb_fused = np.concatenate([rgb_dict[piece_name] for piece_name in xyz_dict.keys()], axis=0)
        normals_fused = np.concatenate([normals_dict[piece_name] for piece_name in xyz_dict.keys()], axis=0)
        normals_fused = np.random.random((xyz_fused.shape[0], 3))
        pcd_fused = BasicPointCloud(points=xyz_fused, colors=rgb_fused, normals=normals_fused)
        ply_path = os.path.join(exp_path, f"ttgs_fused_points3d.ply")
        plydata = storePly(ply_path, xyz_fused,rgb_fused*255, wo_write=False)  # the points3d.ply is not used at all, try not touch the src dataset

        for piece_name in xyz_dict.keys():
            xyz = xyz_dict[piece_name]
            rgb = rgb_dict[piece_name]
            normals = normals_dict[piece_name]
            
            normals = np.random.random((xyz.shape[0], 3))
            pcd_i = BasicPointCloud(points=xyz, colors=rgb, normals=normals)
            if piece_name == 'tissue':
                ply_path = os.path.join(exp_path, f"ttgs_{piece_name}_{tissue_init_mode}_points3d.ply")
            else:
                ply_path = os.path.join(exp_path, f"ttgs_{piece_name}_{tool_init_mode}_points3d.ply")
            plydata = storePly(ply_path, xyz,rgb*255, wo_write=False)  # the points3d.ply is not used at all, try not touch the src dataset
            try:
                pcd_i = fetchPly(path = None, plydata = plydata)
            except:
                pcd_i = None
            pcd_dict[piece_name] = pcd_i
        
    maxtime = endo_dataset.get_maxtime()
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=maxtime,
                           # extend                           
                           scene_metadata=scene_metadata,
                           cam_metadata=cam_metadata,
                           point_cloud_dict = pcd_dict,
                           loaded_obj_names = endo_dataset.obj_tracklets.keys(),
                           )

    return scene_info
    


sceneLoadTypeCallbacks = {
    "endonerf": readEndoNeRFInfo,
}
