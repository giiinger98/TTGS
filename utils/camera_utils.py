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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

 
















WARNED = False

# def loadCam(args, id, cam_info, resolution_scale):

#     return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
#                   FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
#                   image=cam_info.image, mask=None, gt_alpha_mask=None, depth=None,
#                   image_name=cam_info.image_name, uid=id, data_device=args.data_device, 
#                   time = cam_info.time)

# def cameraList_from_camInfos(cam_infos, resolution_scale, args):
#     camera_list = []

#     for id, c in enumerate(cam_infos):
#         camera_list.append(loadCam(args, id, c, resolution_scale))

#     return camera_list


def cameraList_from_camInfos(camera_infos):
    cameras = []
    for cam_info in camera_infos:
        # image = PILtoTorch(cam_info.image, resolution, resize_mode=Image.BILINEAR)[:3, ...]
        # masks = loadmask(cam_info, resolution, resize_mode=Image.NEAREST)
        # metadata = loadmetadata(cam_info.metadata, resolution)

        from scene.cameras import Camera 


        cam_i = Camera(
                    id=cam_info.uid,

                    colmap_id=cam_info.uid, 
                    R=cam_info.R, 
                    T=cam_info.T, 
                    FoVx=cam_info.FovX, 
                    FoVy=cam_info.FovY,
                    K=cam_info.K, 
                    image=cam_info.image, 
                    mask=cam_info.mask, 
                    image_name=cam_info.image_name, 
                    uid=cam_info.uid, 

                    depth=cam_info.depth, 
                    gt_alpha_mask=cam_info.gt_alpha_mask,
                    data_device=cam_info.data_device, 
                    time=cam_info.time,
                    Znear=cam_info.Znear, Zfar=cam_info.Zfar, 
                    h=cam_info.h, w=cam_info.w,
                    masks = cam_info.masks,

                    metadata=cam_info.metadata,
                    dataset_name_for_different_z = cam_info.dataset_name_for_different_z,
                    depth_sudden_change_mask = cam_info.depth_sudden_change_mask,
                    )


        cameras.append(cam_i)

    return cameras








def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
