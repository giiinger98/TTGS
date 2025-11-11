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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.flexible_deform_model import TissueGaussianModel
from scene.tool_model import ToolModel
from utils.sh_utils import eval_sh
from typing import Union
import numpy as np

def get_final_attr_tissue(pc,viewpoint_camera_time, initial_scales,initial_opacity, initial_xyz = None,initial_rotations = None,
                          arap_sampled_idx = None):
    scales = initial_scales
    opacity = initial_opacity
    means3D = pc.get_xyz if initial_xyz is None else initial_xyz
    rotations = pc._rotation if initial_rotations is None else initial_rotations
    ori_time = torch.tensor(viewpoint_camera_time).to(means3D.device)

    if arap_sampled_idx is None:
        deformation_point = pc._deformation_table
        assert means3D.shape[0] == deformation_point.shape[0],f'{means3D.shape[0]} {deformation_point.shape[0]}'
    else:
        assert isinstance(pc,ToolModel)
        deformation_point = pc._deformation_table[arap_sampled_idx]

    try:
        means3D[deformation_point]
    except:
        deformation_point = torch.gt(torch.ones((means3D.shape[0]),device="cuda"),0)
        assert 0, deformation_point
    if isinstance(pc,TissueGaussianModel):
        means3D_deform, scales_deform, rotations_deform = pc.deformation(means3D[deformation_point], 
                                                                        scales[deformation_point], 
                                                                        rotations[deformation_point],
                                                                        ori_time)
    else:
        assert isinstance(pc,ToolModel)
        means3D_deform, scales_deform, rotations_deform = pc.deformation_tools(means3D[deformation_point], 
                                                                        scales[deformation_point], 
                                                                        rotations[deformation_point],
                                                                        ori_time,
                                                                        arap_sampled_idx=arap_sampled_idx,
                                                                        # disable_tool_fdm_scale=True,
                                                                        # disable_tool_fdm_scale=False,
                                                                        )



    opacity_deform = opacity[deformation_point]
    with torch.no_grad():
        if arap_sampled_idx is None:
            pc._deformation_accum[deformation_point] += torch.abs(means3D_deform - means3D[deformation_point])
        else:
            pc._deformation_accum[arap_sampled_idx] += torch.abs(means3D_deform - means3D[deformation_point])
    #FDM
    means3D_final = torch.zeros_like(means3D)
    rotations_final = torch.zeros_like(rotations)
    scales_final = torch.zeros_like(scales)
    opacity_final = torch.zeros_like(opacity)
    means3D_final[deformation_point] =  means3D_deform
    rotations_final[deformation_point] =  rotations_deform
    scales_final[deformation_point] =  scales_deform
    opacity_final[deformation_point] = opacity_deform
    means3D_final[~deformation_point] = means3D[~deformation_point]
    rotations_final[~deformation_point] = rotations[~deformation_point]
    scales_final[~deformation_point] = scales[~deformation_point]
    opacity_final[~deformation_point] = opacity[~deformation_point]
    
    return means3D_final,rotations_final,scales_final,opacity_final
        

def get_final_attr_tool(ttgs_model,viewpoint_camera,tool_parse_cam_again, include_list):
    for i in include_list:
        assert i in ttgs_model.model_name_id.keys(), f'{include_list} {ttgs_model.model_name_id.keys()}'
    
    if tool_parse_cam_again:
        ttgs_model.set_camera_and_maintain_graph_obj_list(graph_obj_list = include_list, camera = viewpoint_camera)#need the self.viewpoint_camera for proper accesing tool transformation related info...
        ttgs_model.transform_obj_pose(include_list = include_list)

    means3D_final = ttgs_model.get_xyz_obj_only
    rotations_final = ttgs_model.get_rotation_obj_only
    return means3D_final,rotations_final



def ttgs_render(viewpoint_camera,
                 pc : Union[TissueGaussianModel,ToolModel], 
                 pipe,
                 bg_color : torch.Tensor, 
                 scaling_modifier = 1.0, 
                 override_color = None,
                 debug_getxyz_ttgs = False,
                 ttgs_model = None,
                 single_compo_or_list = 'tissue',
                 tool_parse_cam_again = True,
                 vis_img_debug = False,
                 vis_img_debug_title_more = 'trn',
                 vis_img_debug_gt_img = None,
                 trn_iter_dbg = None,
                 overlay_binary_mask = None,

                 ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if isinstance(single_compo_or_list,list):
        assert pc == None
        for name in single_compo_or_list:
            assert (name in ['tissue']) or ('obj_tool' in name),single_compo_or_list
    elif 'obj_tool' in single_compo_or_list:
        assert isinstance(pc,ToolModel) 
    elif 'tissue' ==  single_compo_or_list:
        assert isinstance(pc,TissueGaussianModel)
    else:
        assert 0,single_compo_or_list
    
    if isinstance(single_compo_or_list,list):
        assert ttgs_model != None
        assert isinstance(single_compo_or_list,list)
        opacity_list = []
        scales_list = []
        sh_degree_list = []# extend add to get rid of pc
        shs_feature_list = [] # extend add to get rid of pc
        
        single_compo_or_list_idx = {}
        tgt_rendered_gs_idx = 0
        for gs_compo_name in single_compo_or_list:
            pc_i = getattr(ttgs_model,gs_compo_name)
            single_compo_or_list_idx[gs_compo_name] = [tgt_rendered_gs_idx, tgt_rendered_gs_idx+pc_i.get_xyz.shape[0]-1]
            tgt_rendered_gs_idx += pc_i.get_xyz.shape[0]

            opacity = pc_i._opacity
            scales = pc_i._scaling
            sh_degree = pc_i.active_sh_degree # extend            
            shs_feature = pc_i.get_features  # extend            

            opacity_list.append(opacity)
            scales_list.append(scales)
            sh_degree_list.append(sh_degree) # extend            
            shs_feature_list.append(shs_feature) # extend        


        # init screenspace_points and means2D for saving grad in NDC space
        total_gs_num = torch.vstack(scales_list).shape[0]
        screenspace_points = torch.zeros((total_gs_num,3), dtype=pc_i.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        means2D = screenspace_points
        opacity = torch.vstack(opacity_list)
        scales = torch.vstack(scales_list)

        assert len(np.unique(sh_degree_list))==1,'tisseu and tool sh_degree are both 0 for each compo pc'
        sh_degree = np.unique(sh_degree_list)[0] # extend        
        shs_feature = torch.vstack(shs_feature_list) # extend
        # the activation for tool and tisseu are the same-so we can use the last pc_i to getn the activation
        pc_scaling_activation = pc_i.scaling_activation
        pc_rotation_activation = pc_i.rotation_activation
        pc_opacity_activation = pc_i.opacity_activation
            
    else:
        single_compo_or_list_idx = None

        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        means2D = screenspace_points
        opacity = pc._opacity
        scales = pc._scaling
        sh_degree = pc.active_sh_degree
        shs_feature = pc.get_features # extend        
        pc_scaling_activation = pc.scaling_activation
        pc_rotation_activation = pc.rotation_activation
        pc_opacity_activation = pc.opacity_activation
    
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    if pipe.compute_cov3D_python:
        assert 0,'pc required'
    else:
        cov3D_precomp = None
    if single_compo_or_list == 'tissue':
        assert isinstance(pc,TissueGaussianModel) 
        means3D_final,rotations_final,scales_final,opacity_final = get_final_attr_tissue(pc = pc,
                                                                                         viewpoint_camera_time = viewpoint_camera.time,
                                                                                         initial_scales = scales,
                                                                                         initial_opacity= opacity,
                                                                                         )
    elif not isinstance(single_compo_or_list,list) and 'obj_tool' in single_compo_or_list:
        assert isinstance(pc,ToolModel) 
        assert debug_getxyz_ttgs
        assert ttgs_model != None

        include_list=[single_compo_or_list]
        ttgs_model.set_visibility(include_list)# set the self.include_list for ttgs_model
        means3D_final,rotations_final = get_final_attr_tool(ttgs_model=ttgs_model,
                                                            viewpoint_camera=viewpoint_camera,
                                                            tool_parse_cam_again = tool_parse_cam_again,
                                                            # include_list=[single_compo_or_list],
                                                            include_list = include_list,
                                                            )
        scales_final = scales
        opacity_final = opacity

        if pc.use_fdm_tool:
            means3D_final,rotations_final,scales_final,opacity_final = get_final_attr_tissue(pc = pc,
                                                                                            viewpoint_camera_time = viewpoint_camera.time,
                                                                                            initial_scales = scales_final,
                                                                                            initial_opacity= opacity_final,
                                                                                            initial_rotations=rotations_final,
                                                                                            initial_xyz=means3D_final,
                                                                                            )


    elif isinstance(single_compo_or_list,list):

        assert ttgs_model != None
        means3D_final_list = []
        rotations_final_list = []
        scales_final_list = []
        opacity_final_list = []

        for gs_compo_name in single_compo_or_list:
            pc_i = getattr(ttgs_model,gs_compo_name)
            start_idx, end_idx = single_compo_or_list_idx[gs_compo_name]

            scales_i = scales[start_idx:(end_idx+1)]
            opacity_i = opacity[start_idx:(end_idx+1)]

            if isinstance(pc_i,ToolModel):

                include_list=[gs_compo_name]
                ttgs_model.set_visibility(include_list)# set the self.include_list for ttgs_model
                means3D_final,rotations_final = get_final_attr_tool(ttgs_model=ttgs_model,
                                                                    viewpoint_camera=viewpoint_camera,
                                                                    tool_parse_cam_again = tool_parse_cam_again,
                                                                    include_list= include_list,
                                                                    )
                scales_final = scales_i
                opacity_final = opacity_i
                if pc_i.use_fdm_tool:
                    means3D_final,rotations_final,scales_final,opacity_final = get_final_attr_tissue(pc = pc_i,
                                                                                                    viewpoint_camera_time = viewpoint_camera.time,
                                                                                                    initial_scales = scales_final,
                                                                                                    initial_opacity= opacity_final,
                                                                                                    initial_rotations=rotations_final,
                                                                                                    initial_xyz=means3D_final,
                                                                                                    )


            elif isinstance(pc_i,TissueGaussianModel):

                means3D_final,rotations_final,scales_final,opacity_final = get_final_attr_tissue(pc = pc_i,
                                                                                                 viewpoint_camera_time = viewpoint_camera.time,
                                                                                                 initial_scales = scales_i,
                                                                                                 initial_opacity= opacity_i,
                                                                                                 )
            else:
                assert 0
            means3D_final_list.append(means3D_final)
            rotations_final_list.append(rotations_final)
            scales_final_list.append(scales_final)
            opacity_final_list.append(opacity_final)

        means3D_final = torch.vstack(means3D_final_list)
        rotations_final = torch.vstack(rotations_final_list)
        scales_final = torch.vstack(scales_final_list)
        opacity_final = torch.vstack(opacity_final_list)
    else:
        assert 0,single_compo_or_list

    scales_final = pc_scaling_activation(scales_final)
    rotations_final = pc_rotation_activation(rotations_final)
    opacity_final = pc_opacity_activation(opacity_final)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            assert 0
        else:
            shs = shs_feature #pc.get_features
    else:
        assert 0,'not merget yet for override_color'
        colors_precomp = override_color 
    rendered_image, radii, depth, rendered_alpha, rendered_semantic = rasterizer(
        colors_precomp = colors_precomp,
        cov3D_precomp = cov3D_precomp,
        means2D = means2D,
        shs = shs,
        means3D = means3D_final,
        opacities = opacity_final,
        scales = scales_final,
        rotations = rotations_final,
        )
    
    from utils.scene_utils import vis_torch_img
    if vis_img_debug:
        if isinstance(single_compo_or_list,list):
            window_topic = f'{vis_img_debug_title_more}:compo_{single_compo_or_list}_renderObjsOnce'
        else:
            window_topic = f'{vis_img_debug_title_more}:compo_{single_compo_or_list}_renderObjsSeperately'
        
        if overlay_binary_mask is not None:
            from render import blend_img_with_binary_mask,rgba_to_rgb
            rendered_image_vis = blend_img_with_binary_mask(rendered_image,overlay_binary_mask)
            rendered_image_vis = rgba_to_rgb(rendered_image_vis)
            if vis_img_debug_gt_img != None :
                vis_img_debug_gt_img = blend_img_with_binary_mask(vis_img_debug_gt_img,overlay_binary_mask)
                vis_img_debug_gt_img = rgba_to_rgb(vis_img_debug_gt_img)

        else:
            rendered_image_vis = rendered_image

        # rendered_alpha_vis
        vis_torch_img(rendered_image=rendered_image_vis if vis_img_debug_gt_img == None 
                      else torch.cat([rendered_image_vis,
                                      vis_img_debug_gt_img,
                                      ],dim=2),
                      topic=window_topic,
                      )

    return {"render": rendered_image,
            "depth": depth,
            "alpha": rendered_alpha,
            "semantic": rendered_semantic,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,}, single_compo_or_list_idx, means3D_final


