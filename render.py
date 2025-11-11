#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
import imageio
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import ttgs_render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, FDMHiddenParams
from scene.flexible_deform_model import TissueGaussianModel
from scene.tool_model import ToolModel
from time import time
import open3d as o3d
from utils.graphics_utils import fov2focal
import cv2


to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def rgba_to_rgb(rgba, background=(1.0, 1.0, 1.0)):
    """
    Convert an RGBA image to an RGB image by alpha compositing over a background.

    Args:
        rgba (torch.Tensor): A tensor of shape [4, H, W] with values in [0, 1].
        background (tuple): A tuple of three floats representing the RGB background color.
                            Defaults to white (1, 1, 1).

    Returns:
        torch.Tensor: A tensor of shape [3, H, W] representing the composited RGB image.
    """
    # Ensure the background is a tensor and reshape it to [3, 1, 1]
    bg = torch.tensor(background, dtype=rgba.dtype, device=rgba.device).view(3, 1, 1)
    
    # Separate the foreground RGB and the alpha channel
    rgb_fg = rgba[:3]       # Shape: [3, H, W]
    alpha = rgba[3:4]       # Shape: [1, H, W]
    
    # Compute the composited RGB image using alpha blending
    rgb = rgb_fg * alpha + bg * (1 - alpha)
    
    return rgb

def blend_img_with_binary_mask(image, binary_mask, inverse_mask = False):
    assert len(image.shape) == 3, "Image must be RGB"
    assert len(binary_mask.shape) == 3, "Binary mask must be RGB"
    # --- Process the binary mask ---
    binary_mask = binary_mask.squeeze(0).float()  # Now [H, W]
    if inverse_mask:
        binary_mask = 1 - binary_mask
    blue_color = torch.stack([
        torch.zeros_like(binary_mask),  # Red
        torch.zeros_like(binary_mask),  # Green
        torch.ones_like(binary_mask)    # Blue
    ], dim=0)
    mask_alpha = binary_mask  # Shape [H, W]
    mask_alpha = mask_alpha.unsqueeze(0)*0.2  # Now shape [1, H, W]
    alpha_channel = torch.ones_like(image[0:1])  # Fully opaque
    image_rgba = torch.cat([image, alpha_channel], dim=0)
    blended_rgb = image_rgba[:3] * (1 - mask_alpha) + blue_color * mask_alpha
    blended_image = torch.cat([blended_rgb, image_rgba[3:4]], dim=0)
    return blended_image
def ttgs_render_offline(view, gaussians, pipeline, background,
                        controller=None, ttgs_cfg=None,
                        offline_target='tissue',
                        inverse_mask = False,
                        overlay_mask = True,
                        given_mask = None,
                        ):
    '''
    offline render func: call base ttgs_render

    render 1 img for either certain gs model or ttgs controoler
    support offline_target to be a list to render as a whole
    use controller None or not to decide which way'''
    sub_gs_model=None

    if controller==None:    
        assert offline_target == 'tissue'
        rendering,_,means3D_final = ttgs_render(view, gaussians, pipeline, background,
                    single_compo_or_list=offline_target,
                    vis_img_debug = pipeline.dbg_vis_render,
                    vis_img_debug_title_more = 'offline',
                    vis_img_debug_gt_img = view.original_image.cuda(), 
                    overlay_binary_mask = (~given_mask.cuda().bool() if inverse_mask else given_mask.cuda().bool()) if overlay_mask else None,

                    )
        single_compo_or_list_idx = None
        
    else:
        assert ttgs_cfg!= None 
        controller.set_visibility(include_list=list(set(controller.model_name_id.keys()) ))
        for model_name in controller.model_name_id.keys():
            if controller.get_visibility(model_name=model_name):
                if offline_target == model_name:
                    sub_gs_model = None
                    sub_gs_model = getattr(controller, model_name)
                    break
                elif isinstance(offline_target,list):
                    sub_gs_model = None
                else:
                    pass
        if isinstance(offline_target,list):
            assert sub_gs_model==None
        else:
            assert sub_gs_model!=None
        rendering,single_compo_or_list_idx,means3D_final= ttgs_render(view, sub_gs_model, ttgs_cfg.render, background,
                                debug_getxyz_ttgs = True,
                                ttgs_model = controller,
                                single_compo_or_list=offline_target,
                                vis_img_debug = pipeline.dbg_vis_render,
                                vis_img_debug_title_more = 'offline',
                                vis_img_debug_gt_img = view.original_image.cuda(),
                                overlay_binary_mask = (~given_mask.cuda().bool() if inverse_mask else given_mask.cuda().bool()) if overlay_mask else None,
                                )


    return rendering, means3D_final, sub_gs_model, single_compo_or_list_idx



def compute_external_construct_pcd(model_path, name, views, 
    crop_size=0,
    append_target_dir_name = '',
    external_img_dir = None,
    external_depth_dir = None
    ):
    render_images = []
    render_depths = []
    mask_list = []  
 
    assert os.path.exists(external_img_dir),external_img_dir
    assert os.path.exists(external_depth_dir),external_depth_dir
    external_img_list = sorted(os.listdir(external_img_dir))
    external_depth_list = sorted(os.listdir(external_depth_dir))
    assert len(external_img_list)==len(external_depth_list)
    assert len(views) == len(external_img_list),f'{len(views)},{len(external_img_list)}---views need to be video all frames,as the external are fullset--to aligh the idx with each other'
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        read_depth_as_torch_tensor = lambda img_path:torch.tensor(cv2.imread(img_path,cv2.IMREAD_UNCHANGED)).float()
        img_id = idx#int(view.id) 
        print('read img:',img_id,len(external_img_list),len(views))
        img_path = os.path.join(external_img_dir,external_img_list[img_id])
        depth_path = os.path.join(external_depth_dir,external_depth_list[img_id])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = torch.Tensor(img).permute(2,0,1).float()/255
        render_images.append(img)
        render_depths.append(read_depth_as_torch_tensor(depth_path))
    assert len(render_images)==len(views)

    FoVy, FoVx, height, width = view.FoVy, view.FoVx, view.image_height, view.image_width
    focal_y, focal_x = fov2focal(FoVy, height), fov2focal(FoVx, width)
    camera_parameters = (focal_x, focal_y, width, height)
    if 'P2_' in external_img_dir:
        crop_size = 0
    print(len(render_depths),len(render_images))
    print('recon file name:', name)
    reconstruct_point_cloud(render_images, 
                            mask_list, 
                            render_depths, 
                            camera_parameters, 
                            os.path.join(model_path, f'recons_{append_target_dir_name}_{name}'),
                            crop_size,
                            skip_clip='tool' in append_target_dir_name and 'tissue' not in append_target_dir_name,
                            apply_mask='tool' in append_target_dir_name and 'tissue' not in append_target_dir_name,
                            )
    print('recon done for external:',external_img_dir)
    import sys
    sys.exit(0)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background,\
    no_fine, 
    compute_render_fps=False, 
    reconstruct=False, 
    crop_size=0,
    controller=None, ttgs_cfg = None,
    offline_target=None,
    draw_trajectories=True,
    traj_life_span = 20,
    num_trajectory_points = 50,
    traj_start_frame_id = 15,
    traj_end_frame_id = 60,
    use_which_mask_for_supervise = 'xmem',
    all_tool_objs_name = [],
    append_target_dir_name = '',
    toward_metric_computation = False,
    use_external_rather_render = False,
    external_img_dir = None,
    external_depth_dir = None

    ):
    assert use_which_mask_for_supervise in ['xmem','default']
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"Renders{append_target_dir_name}")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"Depth{append_target_dir_name}")
    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    # Add trajectory visualization if requested
    plot_for_each = True
    do_faster = False
    do_faster = True
    aligh_gt_around_render = True
    overlay_mask_on_color_img = True
    save_instance_masks = False
    if toward_metric_computation:
        plot_for_each = False
        draw_trajectories = False
        do_faster = False
        # reconstruct = False
        aligh_gt_around_render = False
        overlay_mask_on_color_img = False
        save_instance_masks = True
        compute_render_fps = True if 'name' in ['test'] else False

    render_images = []
    render_depths = []
    gt_list = []
    gt_depths = []
    mask_list = []  
    means3D_final_list = []
    traj_bg_img = {}
    if draw_trajectories:
        from unit_scripts.traj_related import compute_model_trajectory,create_trajectory_overlay,blend_trajectory_overlay,farthest_point_sample, random_point_sample
        assert traj_end_frame_id>0
    if save_instance_masks:
        view_i = views[0]
        dict_of_obj_instance_list = {}
        for obj in view_i.masks_dict_keys:
            dict_of_obj_instance_list[obj] = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        stage = 'coarse' if no_fine else 'fine'
        if name in ["train", "test", "video"]:
            gt = view.original_image[0:3, :, :]
            gt_list.append(gt)
            #////////////
            if save_instance_masks:
                for obj_i in view.masks_dict_keys:
                    mask_i = getattr(view, f'{obj_i}').bool()#.to(merged_mask.device)
                    dict_of_obj_instance_list[obj_i].append(mask_i)
            mask = getattr(view, f'raw_tissue').bool()#used for overlay
            mask_list.append(mask)#used for overlay
            gt_depth = view.original_depth
            gt_depths.append(gt_depth)

        rendering, means3D_final, sub_gs_model, single_compo_or_list_idx = ttgs_render_offline(view, gaussians, pipeline, background,
                        controller=controller, 
                        ttgs_cfg=ttgs_cfg,
                        offline_target=offline_target,
                        inverse_mask=True,
                        # overlay_mask = True,
                        overlay_mask=overlay_mask_on_color_img,
                        # given_mask = None,
                        given_mask = mask,
                        )
        
        render_depths.append(rendering["depth"].cpu())
        render_images.append(rendering["render"].cpu())
        if draw_trajectories:
            means3D_final_list.append(means3D_final)
            if idx == traj_start_frame_id:
                traj_bg_img[f'STARTIDX{idx}'] = rendering["render"]
            if idx == traj_end_frame_id:
                traj_bg_img[f'ENDIDX{idx}'] = rendering["render"]
            # if idx == traj_start_frame_id:
            if idx == 0:
                if controller is None and isinstance(gaussians, TissueGaussianModel):
                    cur_pts = gaussians._xyz.detach()
                    pts_idx = farthest_point_sample(cur_pts[None], num_trajectory_points)[0]

                elif controller is not None:
                    if isinstance(offline_target, list):
                        pts_idx_all = []
                        for model_name in offline_target:
                            model = getattr(controller, model_name)
                            if isinstance(model, (TissueGaussianModel, ToolModel)):
                                # make the points order of various components are consistent with the means3d
                                cur_pts_i = model._xyz.detach()
                                pts_idx_i = farthest_point_sample(cur_pts_i[None], num_trajectory_points)[0]
                                pts_idx_i = single_compo_or_list_idx[model_name][0] + pts_idx_i
                                pts_idx_all.extend(pts_idx_i)
                        pts_idx = torch.tensor(pts_idx_all)
                    else:
                        cur_pts = sub_gs_model._xyz.detach()
                        pts_idx = farthest_point_sample(cur_pts[None], num_trajectory_points)[0]


    if draw_trajectories:
        total_frames_num = len(means3D_final_list)
        traj_pts = compute_model_trajectory(
            view = view,
            pts_idx=pts_idx,
            # cur_pts = cur_pts,
            means3D_final_list = means3D_final_list,)#N,T,2


        if plot_for_each:
            for idx,rendered_img in enumerate(render_images):
                start_slice = 0
                end_slice = (idx+1)
                if traj_life_span>0:
                    if end_slice-start_slice>traj_life_span:
                        start_slice = idx+1-traj_life_span
                else:
                    assert traj_life_span== -1
                traj_pts_plot = traj_pts[:,start_slice:end_slice]

                # means3D_final
                traj_overlay = create_trajectory_overlay(
                    traj_pts_plot, 
                    view.image_height, 
                    view.image_width, 
                    num_trajectory_points
                )
                # Blend trajectory overlay with rendering
                pts_num,frames,_ = traj_pts_plot.shape
                assert pts_num%num_trajectory_points==0,f'pts_num:{pts_num} num_trajectory_points:{num_trajectory_points}'
                # for bg_img_name, bg_img in traj_bg_img.items():
                blend_traj_overlay = blend_trajectory_overlay(rendered_img, traj_overlay)
                tgt_dir = os.path.join(os.path.dirname(render_path),f'Traj_pts{num_trajectory_points}_{append_target_dir_name}')
                os.makedirs(tgt_dir,exist_ok=True)
                tgt_path = os.path.join(tgt_dir,f"{name}_f{idx}.png")
                torchvision.utils.save_image(blend_traj_overlay,tgt_path)   

        if traj_end_frame_id>0 and traj_end_frame_id<len(means3D_final_list):

            traj_pts = traj_pts[:,traj_start_frame_id:traj_end_frame_id+1]
            # means3D_final
            traj_overlay = create_trajectory_overlay(
                traj_pts, 
                view.image_height, 
                view.image_width, 
                num_trajectory_points
            )
            
            # Blend trajectory overlay with rendering
            pts_num,frames,_ = traj_pts.shape
            assert frames == int(traj_end_frame_id - traj_start_frame_id + 1)
            
            for bg_img_name, bg_img in traj_bg_img.items():
                blend_traj_overlay = blend_trajectory_overlay(bg_img, traj_overlay)

                tgt_path = os.path.join(os.path.dirname(render_path), f"{name}_f{frames}_pts{pts_num}_{bg_img_name}_{append_target_dir_name}.png")
                torchvision.utils.save_image(blend_traj_overlay,tgt_path)            
                print("Save traj in" ,tgt_path)

    if compute_render_fps:
        test_times = 20 #to compute the avg time
        test_times = 1 # quick test
        for i in range(test_times):
            for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
                if idx == 0 and i == 0:
                    time1 = time()
                stage = 'coarse' if no_fine else 'fine'
                rendering_test,_,sub_gs_model_test, single_compo_or_list_idx_test = ttgs_render_offline(view, gaussians, pipeline, background,
                                controller=controller, 
                                ttgs_cfg=ttgs_cfg,
                                offline_target=offline_target,
                                inverse_mask=True,
                                # overlay_mask = True,
                                overlay_mask=overlay_mask_on_color_img,
                                )
        time2=time()
        if test_times>0:
            print('//////////////////////////')
            print(f"{model_path} for {test_times} rounds:")
            print(f"avg FPS for {name} with {test_times} times:",(len(views)-1)*test_times/(time2-time1))

    fused_images = []
    count = 0
    if len(render_images) != 0:
        assert len(render_images) == len(gt_list),f'{len(render_images)},{len(gt_list)}'
        assert len(render_images) == len(mask_list)
        for image, gt_image, binary_mask in zip(render_images, gt_list,mask_list):

            blended_image = blend_img_with_binary_mask(image, binary_mask, inverse_mask=True)
            blended_gt_image = blend_img_with_binary_mask(gt_image, binary_mask, inverse_mask=True)
            fused_image = torch.cat([blended_image, blended_gt_image], dim=2)

            fused_images.append(fused_image)
            if aligh_gt_around_render:
                torchvision.utils.save_image(fused_image,os.path.join(render_path, '{0:05d}'.format(count) + ".png"))
            else:
                torchvision.utils.save_image(image,os.path.join(render_path, '{0:05d}'.format(count) + ".png"))

            count +=1


    count = 0
    if len(render_depths) != 0:
        for depth,gt_depth in tqdm(zip(render_depths,gt_depths)):
            depth = np.clip(depth.cpu().squeeze().numpy().astype(np.uint8), 0, 255)
            gt_depth = np.clip(gt_depth.cpu().squeeze().numpy().astype(np.uint8), 0, 255)
            fused_depth = np.concatenate([depth, gt_depth], axis=1)
            if aligh_gt_around_render:
                cv2.imwrite(os.path.join(depth_path, '{0:05d}'.format(count) + ".png"), fused_depth)
            else:
                cv2.imwrite(os.path.join(depth_path, '{0:05d}'.format(count) + ".png"), depth)

            count += 1

    if not do_faster:
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"gt_images")
        makedirs(gts_path, exist_ok=True)

        count = 0
        if len(gt_list) != 0:
            for image in tqdm(gt_list):
                torchvision.utils.save_image(image, os.path.join(gts_path, '{0:05d}'.format(count) + ".png"))
                count+=1

        if save_instance_masks:
            for mask_name, mask_list in dict_of_obj_instance_list.items(): 
                masks_path_i = os.path.join(model_path, name, "ours_{}".format(iteration), f'masks_{mask_name}')
                makedirs(masks_path_i, exist_ok=True)
                if len(mask_list) != 0:
                    count = 0
                    for mask_i in tqdm(mask_list):
                        mask_i = mask_i.clamp(0, 1)
                        # Convert to uint8 (0-255)
                        mask_i = (mask_i * 255).to(torch.uint8)
                        torchvision.utils.save_image(mask_i.float() / 255.0, os.path.join(masks_path_i, '{0:05d}'.format(count) + ".png"))
                        count +=1

    fps=5
    if fused_images != []:
        render_array = torch.stack(fused_images, dim=0).permute(0, 2, 3, 1)
        render_array = (render_array*255).clip(0, 255).cpu().numpy().astype(np.uint8) # BxHxWxC
        imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), f'fused_video{append_target_dir_name}.mp4'), render_array, fps=fps, quality=8)


    if not do_faster:

        render_array = torch.stack(render_images, dim=0).permute(0, 2, 3, 1)
        render_array = (render_array*255).clip(0, 255).cpu().numpy().astype(np.uint8) # BxHxWxC
        imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), f'video{append_target_dir_name}.mp4'), render_array, fps=fps, quality=8)

        if gt_list!=[]:
            gt_array = torch.stack(gt_list, dim=0).permute(0, 2, 3, 1)
            gt_array = (gt_array*255).clip(0, 255).cpu().numpy().astype(np.uint8)
            imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'gt_video.mp4'), gt_array, fps=fps, quality=8)
                        
    FoVy, FoVx, height, width = view.FoVy, view.FoVx, view.image_height, view.image_width
    focal_y, focal_x = fov2focal(FoVy, height), fov2focal(FoVx, width)
    camera_parameters = (focal_x, focal_y, width, height)

    if reconstruct:
        print('file name:', name)
        gt_depths =  [gt_depth.to(render_depths[0].dtype) for gt_depth in gt_depths]
        reconstruct_point_cloud(render_images, 
                                mask_list, 
                                render_depths, 
                                # gt_depths, 
                                camera_parameters, 
                                os.path.join(model_path, f'recons_{append_target_dir_name}_{name}'),
                                crop_size,
                                skip_clip='tool' in append_target_dir_name and 'tissue' not in append_target_dir_name,
                                apply_mask='tool' in append_target_dir_name and 'tissue' not in append_target_dir_name,
                                )


def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, 
                skip_train : bool, skip_test : bool, skip_video: bool, 
                reconstruct_train: bool, reconstruct_test: bool, reconstruct_video: bool,
                ttgs_cfg = None,controller=None,
                offline_target='tissue',
                append_target_dir_name = '',
                toward_metric_computation = False,
                ):
    with torch.no_grad():
        scene = Scene(dataset)
        if controller is None:
            gaussians = TissueGaussianModel(dataset.sh_degree, hyperparam)
            scene.gs_init(gaussians_or_controller=gaussians, load_iteration=iteration,
                          reset_camera_extent=dataset.camera_extent)
        else:
            scene.gs_init(gaussians_or_controller=controller, load_iteration=iteration,
                          reset_camera_extent=dataset.camera_extent)
            gaussians = None


        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), 
                       gaussians, pipeline, background, False, compute_render_fps = False, reconstruct=reconstruct_train,
                       ttgs_cfg=ttgs_cfg,controller=controller,
                       offline_target=offline_target,

                       use_which_mask_for_supervise='default' if controller==None else 'xmem',
                       all_tool_objs_name = [] if controller==None else controller.all_tool_objs_name,
                        append_target_dir_name = append_target_dir_name,
                        toward_metric_computation = toward_metric_computation,

                       )
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), 
                       gaussians, pipeline, background, False, compute_render_fps = True,reconstruct=reconstruct_test, crop_size=20,
                       ttgs_cfg=ttgs_cfg,controller=controller,
                       offline_target=offline_target,

                       use_which_mask_for_supervise='default' if controller==None else 'xmem',
                       all_tool_objs_name = [] if controller==None else controller.all_tool_objs_name,
                        append_target_dir_name = append_target_dir_name,
                        toward_metric_computation = toward_metric_computation,
                       )
        if not skip_video:
            render_set(dataset.model_path,"video",scene.loaded_iter, scene.getVideoCameras(),
                       gaussians,pipeline,background, False, 
                       compute_render_fps=False, 
                       reconstruct=reconstruct_video, 
                       crop_size=20,
                       ttgs_cfg=ttgs_cfg,controller=controller,
                       offline_target=offline_target,
                       
                       use_which_mask_for_supervise='default' if controller==None else 'xmem',
                       all_tool_objs_name = [] if controller==None else controller.all_tool_objs_name,
                        append_target_dir_name = append_target_dir_name,
                        toward_metric_computation = toward_metric_computation,
                       )



def reconstruct_point_cloud(images, masks, depths, camera_parameters, name, crop_left_size=0,skip_clip = False, apply_mask = False):
    import cv2
    import copy
    output_frame_folder = name#os.path.join("reconstruct", name)
    os.makedirs(output_frame_folder, exist_ok=True)
    frames = np.arange(len(images))
    focal_x, focal_y, width, height = camera_parameters
    if crop_left_size > 0:
        width = width - crop_left_size
        height = height - crop_left_size//2
    for i_frame in frames:
        rgb_tensor = images[i_frame]
        rgb_np = rgb_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).contiguous().to("cpu").numpy()
        depth_np = depths[i_frame].cpu().numpy()

        if len(depth_np.shape) == 3:
            depth_np = depth_np[0]
        # depth_np = depth_np.squeeze(0)
        if crop_left_size > 0:
            rgb_np = rgb_np[:, crop_left_size:, :]
            depth_np = depth_np[:, crop_left_size:]
            rgb_np = rgb_np[:-crop_left_size//2, :, :]
            depth_np = depth_np[:-crop_left_size//2, :]
        rgb_new = copy.deepcopy(rgb_np)

        if apply_mask:
            mask = masks[i_frame]
            mask = mask.squeeze(0).cpu().numpy().astype(np.bool)
            mask = ~mask
            mask = mask[:, crop_left_size:]
            mask = mask[:-crop_left_size//2, :]
            depth_np[mask == 0] =np.nan #0
            rgb_new[mask ==0] = np.asarray([0,0,0]) 


        depth_smoother = (32, 64, 32) # (128, 64, 64) #[24, 64, 32]
        depth_np = cv2.bilateralFilter(depth_np, depth_smoother[0], depth_smoother[1], depth_smoother[2])
                
        skip_clip = True
        if not skip_clip:
            print('reconstruct use the most strict depther_filter')
            close_depth = np.percentile(depth_np[depth_np!=0], 5)
            inf_depth = np.percentile(depth_np, 95)
            depth_np = np.clip(depth_np, close_depth, inf_depth)
        
        rgb_im = o3d.geometry.Image(rgb_new.astype(np.uint8))
        depth_im = o3d.geometry.Image(depth_np)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_im, depth_im, convert_rgb_to_intensity=False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(int(width), int(height), focal_x, focal_y, width / 2, height / 2),
            project_valid_depth_only=True
        )
        o3d.io.write_point_cloud(os.path.join(output_frame_folder, 'frame_{}.ply'.format(i_frame)), pcd)
        print('reconsuted saved in ',os.path.join(output_frame_folder, 'frame_{}.ply'.format(i_frame)))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = FDMHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--reconstruct_train", action="store_true")
    parser.add_argument("--reconstruct_test", action="store_true")
    parser.add_argument("--reconstruct_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--offline_target", type=str, 
                        nargs='*', 
                        default=[],
                        help="List of targets to render. Default empty list is rendering all, else choose spercific ones: ['tissue', 'obj_tool1', 'obj_tool2']")
    
    # it will load the cfg_args saving the args when train
    # args = get_combined_args(parser,insist=True)
    allow_multiple = True
    args_list = get_combined_args(parser,insist=True, allow_multiple=allow_multiple)
    for args_i in args_list:
        args = args_i
        print("Rendering ", args.model_path)
        toward_metric_computation = True
        if args.method == 'ttgs':
            render_with_ttgs= True
            offline_target = args.offline_target
            assert isinstance(offline_target, list)
            offline_target = sorted(offline_target)
            append_target_dir_name = '_'.join(offline_target)
        elif args.method == 'deform3dgs':
            render_with_ttgs= False
            append_target_dir_name = ''
        else:
            assert 0, args.method
        
        args.dbg_vis_render = False
        #//////////////////////////////////////////////////////////
        #check if args contain certain variable 
        #during develop extend new variable---render old model
        if not hasattr(args, 'inited_pcd_noise_removal'):
            setattr(args, 'inited_pcd_noise_removal', False)
            setattr(args, 'supervise_depth_noise_ignore_tgt', [])
        safe_state(args.quiet)  
        #//////////////////////////////////////////////////////////////

        if not render_with_ttgs:
            #use the FDM model to render
            render_sets(model.extract(args), hyperparam.extract(args), args.iteration, 
                pipeline.extract(args), 
                args.skip_train, args.skip_test, args.skip_video,
                args.reconstruct_train,
                reconstruct_test = args.reconstruct_test,
                reconstruct_video = args.reconstruct_video,
                append_target_dir_name = append_target_dir_name,
                toward_metric_computation = toward_metric_computation,
                )
        else:
            #use the ttgs model to render
            exp_time_cfg_file_name = 'configs/config_000000.yaml'
            cfg_path = os.path.join(args.model_path,exp_time_cfg_file_name)
            assert os.path.exists(cfg_path),f'not saved cfg during traing? {args.configs}'
            from config.yacs import load_cfg
            with open(cfg_path, 'r') as f:
                ttgs_cfg = load_cfg(f)

            if not hasattr(ttgs_cfg.model, 'disable_tool_fdm_scale'):
                setattr(ttgs_cfg.model, 'disable_tool_fdm_scale',False)
            if not hasattr(ttgs_cfg.optim, 'tool_percent_dense'):
                setattr(ttgs_cfg.optim, 'tool_percent_dense',0.0)
            if not hasattr(ttgs_cfg.model, 'basis_type'):
                setattr(ttgs_cfg.model, 'basis_type','gaussian')

            from scene.tt_gaussian_model import TTGaussianModel
            scene = Scene(model.extract(args),load_other_obj_meta=True,new_cfg=ttgs_cfg)
            controller = TTGaussianModel(metadata=scene.getSceneMetaData(),
                                        new_cfg=ttgs_cfg)#nn.module instance

            #/////
            trained_objs = controller.candidate_model_names['obj_model_cand']+controller.candidate_model_names['tissue_model']
            trained_objs = controller.compo_all_gs_ordered_renderonce
            if offline_target==[]:
                offline_target = trained_objs.copy()#controller.all_tool_objs_name#['tissue','obj_tool1','obj_tool2']
            if sorted(offline_target) != sorted(trained_objs):
                append_target_dir_name = ''

            render_sets(model.extract(args), hyperparam.extract(args), args.iteration, 
                pipeline.extract(args), 
                args.skip_train, args.skip_test, args.skip_video,
                args.reconstruct_train,
                reconstruct_test = args.reconstruct_test,
                reconstruct_video = args.reconstruct_video,
                ttgs_cfg=ttgs_cfg,
                controller=controller,
                offline_target=offline_target,
                append_target_dir_name = append_target_dir_name,
                toward_metric_computation=toward_metric_computation,

                )
        print('Finished for',args.model_path)

