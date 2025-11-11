#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import random
import os 
import torch
from random import randint
from gaussian_renderer import ttgs_render

import sys
from scene import  Scene
from scene.flexible_deform_model import TissueGaussianModel
from scene.tt_gaussian_model import TTGaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from arguments import FDMHiddenParams as ModelHiddenParams
from utils.timer import Timer
import torch.nn.functional as F

from utils.scene_utils import render_training_image
from scene.cameras import Camera
from utils.loss_utils import ssim
from metrics import cal_lpips

from train import training_report
from utils.image_utils import save_img_torch, visualize_depth_numpy



to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training_ttgsmodel(args,
                        ):
    


    dbg_print = args.dbg_print
    remain_redundant_default_param = args.remain_redundant_default_param

    from config.argsgroup2cn import perform_args2cfg

    cfg, others = perform_args2cfg(args,
                                    remain_redundant = remain_redundant_default_param,
                                    dbg_print = dbg_print,
                                    )
    eval_stree_param,train_stree_param,opt_stree_param,\
            mod_stree_param,data_stree_param,render_stree_param,viewer_stree_param,\
                other_param_dict = others

    os.makedirs(cfg.trained_model_dir,exist_ok=True)
    os.makedirs(cfg.point_cloud_dir,exist_ok=True)
    from train import prepare_output_and_logger
    tb_writer = prepare_output_and_logger(model_path=cfg.expname, write_args=args)
    timer = Timer()
    load_other_obj_meta=True
    load_pcd_dict_in_sceneinfo=True
    # ttgs reuse the Scene function of deform3dgs (only break it down)
    scene = Scene(mod_stree_param,
                  load_other_obj_meta=load_other_obj_meta,
                  new_cfg=cfg,
                  load_pcd_dict_in_sceneinfo=load_pcd_dict_in_sceneinfo,
                  )
    
    controller = TTGaussianModel(metadata=scene.getSceneMetaData(),
                                 new_cfg=cfg)
    scene.gs_init(gaussians_or_controller=controller,
                  reset_camera_extent=mod_stree_param.camera_extent)
    timer.start()




    scene_reconstruction_ttgs(cfg = cfg, controller = controller,
                               scene = scene, tb_writer = tb_writer,
                               render_stree_param_for_ori_train_report = render_stree_param,
                               )



def compute_more_metrics(gt_image,
                         renderOnce,image_all,
                         cfg,
                         tissue_mask,tool_mask,more_to_log,
                            use_ema = True,
                            ema_psnr_for_log_tissue = None,
                            ema_psnr_for_log_tool = None,
                            dir_append = '',
                            compute_ssim = False,
                            compute_lpips = False,
                            renderSeperate_image_dict = {},
                            tool_masks_dict = {},
                            detail_log_tool = False,
                            ):
    assert isinstance(renderSeperate_image_dict,dict)
    if not renderOnce:
        assert renderSeperate_image_dict!= {}
    psnr_weight_ori = 0.6 if use_ema else 0
    psnr_weight_current = 1-psnr_weight_ori
    log_psnr_name = 'ema_psnr' if use_ema else 'crt_psnr'
    log_ssim_name = 'ssim'
    log_lpips_name = 'lpips'
    if use_ema:
        assert ema_psnr_for_log_tissue!= None
        assert ema_psnr_for_log_tool!= None

    if renderOnce:
        image_tissue = image_all
    else:
        image_tissue = renderSeperate_image_dict['tissue']
    if cfg.model.nsg.include_tissue:
        ema_psnr_for_log_tissue = psnr_weight_current * psnr(image_tissue, gt_image, tissue_mask).mean().float() 
        + psnr_weight_ori * ema_psnr_for_log_tissue
        more_to_log[f'tissue/{log_psnr_name}{dir_append}'] = ema_psnr_for_log_tissue
        if compute_ssim:
            ssim_for_log_tissue = ssim(image_tissue.to(torch.double), gt_image.to(torch.double), mask = tissue_mask).mean().float() 
            more_to_log[f'tissue/{log_ssim_name}{dir_append}'] = ssim_for_log_tissue
        if compute_lpips:
            lpips_for_log_tissue = cal_lpips((image_tissue*tissue_mask.unsqueeze(0)).to(torch.float32), 
                                         (gt_image*tissue_mask.unsqueeze(0)).to(torch.float32), 
                                         ).mean().float() 
            more_to_log[f'tissue/{log_lpips_name}{dir_append}'] = lpips_for_log_tissue
    else:
        assert 0,'alwasy include tissue'

    if cfg.model.nsg.include_obj:
        tool_masks_dict = tool_masks_dict
        for tool_name,tool_mask_i in tool_masks_dict.items():
            if not detail_log_tool:
                if tool_name != 'tool':
                    continue
            if not renderOnce:
                if tool_name=='tool':
                    continue
                image_tool_i = renderSeperate_image_dict[tool_name] 
            else:
                image_tool_i = image_all

            ema_psnr_for_log_tool = psnr_weight_current * psnr(image_tool_i, gt_image, tool_mask_i).mean().float() 
            + psnr_weight_ori * ema_psnr_for_log_tool
            more_to_log[f'{tool_name}/{log_psnr_name}{dir_append}'] =  ema_psnr_for_log_tool
            if compute_ssim:
                ssim_for_log_tool = ssim(image_tool_i.to(torch.double), gt_image.to(torch.double), mask = tool_mask_i).mean().float() 
                more_to_log[f'{tool_name}/{log_ssim_name}{dir_append}'] = ssim_for_log_tool
            if compute_lpips:
                lpips_for_log_tool = cal_lpips((image_tool_i*tool_mask_i.unsqueeze(0)).to(torch.float32),  
                                        (gt_image*tool_mask_i.unsqueeze(0)).to(torch.float32), 
                                        ).mean().float() 
                more_to_log[f'{tool_name}/{log_lpips_name}{dir_append}'] = lpips_for_log_tool
            
    else:
        assert 0,'alwasy include tool'
    return ema_psnr_for_log_tissue,ema_psnr_for_log_tool,more_to_log

def render_ttgs_n_compute_loss(controller,viewpoint_cam,cfg,training_args,optim_args,
                                renderOnce,
                                debug_getxyz_ttgs,
                                iteration,
                                skip_loss_compute = False,
                                vis_img_debug_title_more = 'trn',
                                tool_parse_cam_again_renderonce = True,
                                arap_rand_viewpoint_cams = [],
                                do_asap_for_tools = False,
                                ):
    '''
    shared to use by train and test during training
    '''
    gt_image = viewpoint_cam.original_image.cuda()
    gt_depth_raw = viewpoint_cam.original_depth.cuda()
    depth_sudden_change_mask = viewpoint_cam.depth_sudden_change_mask.cuda() \
        if viewpoint_cam.depth_sudden_change_mask!=None else None 
    tissue_mask = viewpoint_cam.raw_tissue.cuda().bool()
    tool_masks_dict = {}

    if (iteration - 1) == training_args.debug_from:
        cfg.render.debug = True

    bg_color = [1, 1, 1] if cfg.data.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    loss_dict = dict()
    from utils.loss_utils import l1_loss
    radii_all_compo_adc = {}
    visibility_filters_all_compo_adc = {}
    viewspace_point_tensors_all_compo_adcdict = {}
    model_names_all_compo_adc = []

    Ll1_tissue = torch.tensor(0.).cuda()
    Ll1_tool = torch.tensor(0.).cuda()
    Ll1_tool_i = torch.tensor(0.).cuda()
    depth_loss_tissue = torch.tensor(0.).cuda()
    depth_loss_tool = torch.tensor(0.).cuda()
    depth_loss_tool_i = torch.tensor(0.).cuda()
    reg_loss = torch.tensor(0.).cuda()

    debug_compute_tool_pose_reg = False
    lambda_tool_pose_reg = 0.01
    tool_pose_reg_loss = torch.tensor(0.).cuda()


    # process tissue_mask
    assert cfg.model.process_tissue_mask_trn in [None,'erode']
    assert cfg.model.process_tool_mask_trn in [None,'erode']
    from utils.general_utils import erode_mask_torch

    tissue_mask = erode_mask_torch(masks = tissue_mask,\
                                   kernel_size = 80) if cfg.model.process_tissue_mask_trn == 'erode' else tissue_mask



    compo_all_gs_ordered_renderonce = controller.compo_all_gs_ordered_renderonce

    reg_once_when_seperate_trn = False
    reg_once_when_seperate_freq = 100

    for render_which in controller.all_tool_objs_name:
        tool_mask_obj_i = getattr(viewpoint_cam, f'raw_{render_which}').cuda().bool()
        tool_masks_dict[render_which] = tool_mask_obj_i

    tool_mask = torch.stack(list(tool_masks_dict.values())).any(dim=0)
    tool_masks_dict['tool'] = tool_mask

    if renderOnce:   
        render_pkg_all,compo_all_gs_ordered_idx,_ = ttgs_render(viewpoint_cam, 
                                        None, 
                                        cfg.render, 
                                        background,
                                    debug_getxyz_ttgs=debug_getxyz_ttgs,
                                    ttgs_model=controller,
                                    single_compo_or_list=compo_all_gs_ordered_renderonce,
                                    tool_parse_cam_again = tool_parse_cam_again_renderonce,
                                    vis_img_debug = cfg.render.dbg_vis_render,
                                    vis_img_debug_title_more = vis_img_debug_title_more,
                                    vis_img_debug_gt_img = gt_image,
                                    trn_iter_dbg = iteration,
                                    overlay_binary_mask = ~tissue_mask,
                                    )
        image_all = render_pkg_all["render"]
        depth_all = render_pkg_all["depth"]

        if not skip_loss_compute:
            depth_all = depth_all.unsqueeze(0)
            gt_depth = gt_depth_raw.unsqueeze(0)
            depth_all[depth_all!=0] = 1 / depth_all[depth_all!=0]
            gt_depth[gt_depth!=0] = 1 / gt_depth[gt_depth!=0]

            for render_which in compo_all_gs_ordered_renderonce:
                if render_which == 'tissue':
                    if 'color' in cfg.model.tissue_mask_loss_src:
                        Ll1_tissue = l1_loss(image_all, gt_image, tissue_mask)
                    if 'depth' in cfg.model.tissue_mask_loss_src:
                        if (gt_depth!=0).sum() >= 10:
                            depth_loss_tissue = l1_loss(depth_all, gt_depth, tissue_mask)
                elif render_which.startswith('obj_tool'):
                    tool_mask_obj_i = tool_masks_dict[render_which]
                    if 'color' in cfg.model.tool_mask_loss_src:
                        Ll1_tool_i = l1_loss(image_all, gt_image, tool_mask_obj_i,
                                            )
                        Ll1_tool += Ll1_tool_i
                    if 'depth' in cfg.model.tool_mask_loss_src:
                        if (gt_depth!=0).sum() >= 10:
                            depth_loss_tool_i = l1_loss(depth_all, gt_depth, tool_mask_obj_i)
                            depth_loss_tool += depth_loss_tool_i
                    loss_dict[f'loss/Tool_loss_{render_which}'] = Ll1_tool_i.item()+depth_loss_tool_i.item()
                else:
                    assert 0




    
    else:
    # if not renderOnce:   

        render_pkg_all = {}
        if cfg.model.nsg.include_tissue:
            render_pkg_tissue,_,_ = ttgs_render(viewpoint_cam, controller.tissue, cfg.render, background,
                                            single_compo_or_list='tissue',
                                            tool_parse_cam_again = False,# no need for tissue
                                            vis_img_debug = cfg.render.dbg_vis_render,
                                            vis_img_debug_title_more = vis_img_debug_title_more,
                                            vis_img_debug_gt_img = gt_image,
                                            trn_iter_dbg = iteration,



                                            )
            image_tissue, depth_tissue, viewspace_point_tensor_tissue, visibility_filter_tissue, radii_tissue = \
                render_pkg_tissue["render"], render_pkg_tissue["depth"], render_pkg_tissue["viewspace_points"], \
                    render_pkg_tissue["visibility_filter"], render_pkg_tissue["radii"]
            acc_tissue = torch.zeros_like(depth_tissue)
            if not skip_loss_compute:
                Ll1_tissue = l1_loss(image_tissue, gt_image, tissue_mask)
                Ll1_tissue = (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1_tissue + \
                    optim_args.lambda_dssim * (1.0 - ssim(image_tissue.to(torch.double), \
                                                        gt_image.to(torch.double), mask=tissue_mask))
            
            model_names_all_compo_adc.append('tissue')
            radii_all_compo_adc['tissue'] = radii_tissue
            visibility_filters_all_compo_adc['tissue'] = visibility_filter_tissue
            viewspace_point_tensors_all_compo_adcdict['tissue'] = viewspace_point_tensor_tissue
            render_pkg_all = {
                'tissue':render_pkg_tissue,
                }     
            
        else:
            assert 0,'alwasy include tissue'



        if cfg.model.nsg.include_obj:
            for render_which in controller.candidate_model_names['obj_model_cand']: 
                render_pkg_tool_i,_,_ = ttgs_render(viewpoint_cam, 
                                                getattr(controller,render_which),
                                                cfg.render, background,
                                            debug_getxyz_ttgs=debug_getxyz_ttgs,
                                            ttgs_model=controller,
                                            single_compo_or_list=render_which,
                                            tool_parse_cam_again = tool_parse_cam_again_renderonce,
                                            vis_img_debug = cfg.render.dbg_vis_render,
                                            vis_img_debug_title_more = vis_img_debug_title_more,
                                            trn_iter_dbg = iteration,

                                            )
                image_tool, depth_tool, viewspace_point_tensor_tool, visibility_filter_tool, radii_tool = \
                    render_pkg_tool_i["render"], render_pkg_tool_i["depth"], render_pkg_tool_i["viewspace_points"], \
                        render_pkg_tool_i["visibility_filter"], render_pkg_tool_i["radii"]
                
                tool_mask_obj_i = tool_masks_dict[render_which] 
                if not skip_loss_compute:
                    Ll1_tool_i = l1_loss(image_tool, gt_image, tool_mask_obj_i)
                    Ll1_tool_i = (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1_tool_i \
                        + optim_args.lambda_dssim * (1.0 - ssim(image_tool.to(torch.double), gt_image.to(torch.double), \
                                                                mask=tool_mask_obj_i))
                    Ll1_tool += Ll1_tool_i
                    loss_dict[f'loss/Tool_loss_{render_which}'] = Ll1_tool_i.item()+depth_loss_tool_i.item()

                model_names_all_compo_adc.append(render_which)
                radii_all_compo_adc[render_which] = radii_tool
                visibility_filters_all_compo_adc[render_which] = visibility_filter_tool
                viewspace_point_tensors_all_compo_adcdict[render_which] = viewspace_point_tensor_tool

                render_pkg_all[f'{render_which}'] = render_pkg_tool_i

        compo_all_gs_ordered_idx = None
        image_all = None

    tool_reg_mode = 'discri_v2'
    if tool_reg_mode == 'discri':
        lambda_tool_reg = 1
        reg_since_iter = 700
        lambda_tissue_reg = .01
        reg_freq = 1
    elif tool_reg_mode == 'discri_v2':
        lambda_tool_reg = 0.1
        reg_since_iter = 0
        lambda_tissue_reg = .01
        reg_freq = 1

    if cfg.model.reg_fg_tool or cfg.model.reg_bg_tissue:
        if iteration >= reg_since_iter and iteration % reg_freq == 0:
            render_pkg_tools = {}
            means3D_finals = {}
            renderOnce_tool_only_list = [ i for i in compo_all_gs_ordered_renderonce if i.startswith('obj_tool')]
            if renderOnce:
                render_pkg_tools = {}
                render_pkg_tool_allonce,_,means3D_final_allonce = ttgs_render(viewpoint_cam, 
                                                None,
                                                cfg.render, 
                                            background,
                                            debug_getxyz_ttgs=debug_getxyz_ttgs,
                                            ttgs_model=controller,
                                            single_compo_or_list=renderOnce_tool_only_list,
                                            tool_parse_cam_again = tool_parse_cam_again_renderonce,
                                            vis_img_debug = cfg.render.dbg_vis_render,
                                            vis_img_debug_title_more = f'trn_Reg_all_tool {"".join(renderOnce_tool_only_list)}',
                                            trn_iter_dbg = iteration,
                                            )
            else:
                render_pkg_tools = render_pkg_all

            if cfg.model.reg_fg_tool:
                tool_mask_allonce = torch.zeros_like(tool_masks_dict['tool'])
                for tool_name in renderOnce_tool_only_list:
                    tool_mask_obj_i = tool_masks_dict[tool_name]
                    tool_mask_allonce = torch.logical_or(tool_mask_allonce,tool_mask_obj_i)

                alpha_tool_allonce =  render_pkg_tool_allonce['alpha']
                acc_obj_allonce = torch.clamp(alpha_tool_allonce, min=1e-6, max=1.-1e-6)
                if tool_reg_mode == 'discri':
                    obj_acc_loss_allonce = torch.where(tool_mask_allonce, 
                        -(acc_obj_allonce * torch.log(acc_obj_allonce) +  (1. - acc_obj_allonce) * torch.log(1. - acc_obj_allonce)), 
                        -torch.log(1. - acc_obj_allonce)).mean()
                elif tool_reg_mode == 'discri_v2':
                    obj_acc_loss_allonce = torch.where(tool_mask_allonce, 
                        -(acc_obj_allonce * torch.log(acc_obj_allonce)), 
                        -torch.log(1. - acc_obj_allonce)).mean()
                else:
                    assert 0, tool_reg_mode
                loss_dict['loss/tool_alpha_reg'] = obj_acc_loss_allonce.item()
                reg_loss += lambda_tool_reg * obj_acc_loss_allonce

                if do_asap_for_tools:
                    asap_smple_ratio = 0.01
                    asap_reg_weight = 0.1


                    lambda_arap_landmarks = [ 1e-4,  1e-4,  1e-5,  1e-5,     0]
                    lambda_arap_steps =     [    0,  5000, 10000, 20000, 20001]

                    def landmark_interpolate(landmarks, steps, step, interpolation='log'):
                        stage = (step >= np.array(steps)).sum()
                        if stage == len(steps):
                            return max(0, landmarks[-1])
                        elif stage == 0:
                            return 0
                        else:
                            ldm1, ldm2 = landmarks[stage-1], landmarks[stage]
                            if ldm2 <= 0:
                                return 0
                            step1, step2 = steps[stage-1], steps[stage]
                            ratio = (step - step1) / (step2 - step1)
                            if interpolation == 'log':
                                return np.exp(np.log(ldm1) * (1 - ratio) + np.log(ldm2) * ratio)
                            elif interpolation == 'linear':
                                return ldm1 * (1 - ratio) + ldm2 * ratio
                            else:
                                print(f'Unknown interpolation type: {interpolation}')
                                raise NotImplementedError

                    asap_reg_weight = landmark_interpolate(landmarks=lambda_arap_landmarks, 
                                                           steps=lambda_arap_steps, 
                                                           step=iteration)

                    if arap_rand_viewpoint_cams != []:
                        for piece_name in renderOnce_tool_only_list:
                            arap_loss_value_tool = getattr(controller,piece_name).arap_loss(
                                sample_ratio=asap_smple_ratio,
                                another_cams = arap_rand_viewpoint_cams,
                                ttgs_model = controller,
                            )
                            reg_loss += asap_reg_weight * arap_loss_value_tool
                            print(f'//////////ARAP//{piece_name}/////',arap_loss_value_tool)



    weight_tool = 1.0
    weight_tissue = 1.0

    Ll1 = weight_tissue*Ll1_tissue + weight_tool*Ll1_tool
    depth_loss = weight_tissue*depth_loss_tissue + weight_tool*depth_loss_tool
    loss = Ll1 + depth_loss + reg_loss



    if renderOnce:            
        for model_name,(start_idx,end_idx) in compo_all_gs_ordered_idx.items():
                model_names_all_compo_adc.append(model_name)
                assert end_idx<len(render_pkg_all["radii"]),f'end_idx {end_idx} should be less than {len(render_pkg_all["radii"])}'
                assert len(render_pkg_all["radii"])==len(render_pkg_all["visibility_filter"])
                assert len(render_pkg_all["radii"])==len(render_pkg_all["viewspace_points"])
                radii_all_compo_adc[model_name] = render_pkg_all["radii"][start_idx:(end_idx+1)]
                visibility_filters_all_compo_adc[model_name] = render_pkg_all["visibility_filter"][start_idx:(end_idx+1)]
        assert model_names_all_compo_adc == compo_all_gs_ordered_renderonce,'sanity check'
    else:
        pass

    loss_dict['loss/Tissue_loss'] = Ll1_tissue.item()+depth_loss_tissue.item()
    loss_dict['loss/Tool_loss'] = Ll1_tool.item()+depth_loss_tool.item()
    loss_dict['loss/Color_loss'] = Ll1.item()
    loss_dict['loss/Depth_loss'] = depth_loss.item()
    loss_dict['loss/Total_loss'] = loss.item()

    if skip_loss_compute:
         Ll1, loss = None,None


    return render_pkg_all,compo_all_gs_ordered_idx, Ll1, loss,loss_dict,\
        gt_image,viewpoint_cam.original_depth.cuda(),image_all,tissue_mask,tool_mask,tool_masks_dict,\
        radii_all_compo_adc,visibility_filters_all_compo_adc,model_names_all_compo_adc,\
            viewspace_point_tensors_all_compo_adcdict,visibility_filters_all_compo_adc,\
                model_names_all_compo_adc



 

def scene_reconstruction_ttgs(cfg, controller, scene, tb_writer,
                               render_stree_param_for_ori_train_report = None,
                               debug_getxyz_ttgs = True,
                               ):
    
    # Use TTGS scene_recon function:1) entry for more loss 2) render with ttgs controller...
    training_args = cfg.train
    optim_args = cfg.optim
    data_args = cfg.data
    model_args = cfg.model
    start_iter = 0
    controller.training_setup()
    print(f'Starting from Iteration: {start_iter}')
    from config.argsgroup2cn import save_cfg
    save_cfg(cfg, cfg.model_path, epoch=start_iter)




    compute_more_metrics_flag = True
    monitor_trn_by_render_seperately_freq = 20
    disable_log_in_pts_num = False
    disable_save_model = False


    do_test = model_args.do_test
    tb_report_freq = model_args.tb_report_freq
    test_freq = model_args.test_freq
    assert tb_report_freq == test_freq ,f'avoid waste testing'
    
    renderOnce = model_args.renderOnce
    compo_all_gs_ordered_renderonce = model_args.compo_all_gs_ordered_renderonce
    use_ema_train = model_args.use_ema_train
    use_ema_test = model_args.use_ema_test

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    ema_psnr_for_log_tissue_trn = 0.0
    ema_psnr_for_log_tool_trn = 0.0
    ema_psnr_for_log_tissue_test = 0.0
    ema_psnr_for_log_tool_test = 0.0

    progress_bar = tqdm(range(start_iter, training_args.iterations))
    start_iter += 1



    viewpoint_stack = None
    more_to_logs = {}

    for iteration in tqdm(range(start_iter, training_args.iterations + 1)):
    
        iter_start.record()
        controller.update_learning_rate(iteration)

        if iteration % 500 == 0:
            controller.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        arap_rand_viewpoint_cams = []
        if cfg.model.do_asap_for_tools:
            for _ in range(2):
                try:
                    next_cam = scene.getTrainCameras()[viewpoint_cam.uid+1]
                    prv_cam = scene.getTrainCameras()[viewpoint_cam.uid-1]
                    arap_rand_viewpoint_cams.append(next_cam)
                    arap_rand_viewpoint_cams.append(prv_cam)
                except:
                    print('drop the arap due to missing prv/next for nbr arap')

        render_pkg_all,compo_all_gs_ordered_idx,              Ll1, loss, loss_dict,\
            gt_image, gt_depth, image_all,tissue_mask,tool_mask,tool_masks_dict,\
                radii_all_compo_adc,visibility_filters_all_compo_adc,model_names_all_compo_adc,\
                    viewspace_point_tensors_all_compo_adcdict,visibility_filters_all_compo_adc,\
                        model_names_all_compo_adc = render_ttgs_n_compute_loss(controller,viewpoint_cam,cfg,training_args,optim_args,
                                renderOnce,
                                debug_getxyz_ttgs,
                                iteration,
                                vis_img_debug_title_more='trn',
                                tool_parse_cam_again_renderonce = True,
                                arap_rand_viewpoint_cams = arap_rand_viewpoint_cams,
                                do_asap_for_tools = cfg.model.do_asap_for_tools,
                            )
        loss.backward()
        iter_end.record()

        more_to_log = {}
        
        with torch.no_grad():
            renderSeperate_image_dict = {}
            if not renderOnce:
                for key in render_pkg_all.keys():
                    assert key == 'tissue' or key.startswith('obj_tool'),f'sanity check{key}'
                    extract_key = key
                    assert extract_key in tool_masks_dict.keys() or extract_key=='tissue',f'sanity check{key}'
                    renderSeperate_image_dict[extract_key] = render_pkg_all[key]['render']
            if compute_more_metrics_flag:
                ema_psnr_for_log_tissue_trn,ema_psnr_for_log_tool_trn,more_to_log = \
                    compute_more_metrics(gt_image,renderOnce,image_all,cfg,
                                        tissue_mask=tissue_mask,
                                        tool_mask = tool_mask,
                                        more_to_log = more_to_log,
                                        use_ema=use_ema_train,
                                        ema_psnr_for_log_tissue=ema_psnr_for_log_tissue_trn,
                                        ema_psnr_for_log_tool=ema_psnr_for_log_tool_trn,
                                        dir_append = '_trn',
                                        renderSeperate_image_dict = renderSeperate_image_dict,
                                        tool_masks_dict = tool_masks_dict,
                                        detail_log_tool = True,
                                        )
            else:
                more_to_log=more_to_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Exp": f"{cfg.task}-{cfg.exp_name}", 
                                        "PSNR_tissue": f"{ema_psnr_for_log_tissue_trn:.{4}f}",
                                        "PSNR_tool": f"{ema_psnr_for_log_tool_trn:.{4}f}",
                                        }
                                        )
                progress_bar.update(10)
            if iteration == training_args.iterations:
                progress_bar.close()
            # Log and save
            if (iteration in training_args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                stage =''
                scene.save(iteration,stage =stage)
                if isinstance(scene.gaussians_or_controller, TTGaussianModel):
                    pose_model_root = os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration))
                    pose_model_path = os.path.join(pose_model_root, "pose_model.pth")
                    if scene.gaussians_or_controller.poses_all_objs is not None and not disable_save_model:
                        state_dict = scene.gaussians_or_controller.poses_all_objs.save_state_dict()
                        torch.save(state_dict, pose_model_path)

            debug_update_input_poses = True
            register_corrected_pose_freq = 500
            update_input_poses_until = -1
            if debug_update_input_poses and iteration % register_corrected_pose_freq == 0:
                pose_model = scene.gaussians_or_controller.poses_all_objs 
                if pose_model is not None and cfg.model.use_opt_track:
                    pose_model.save_learned_tool_pose(video_all_Cameras = scene.getVideoCameras(),
                                                      saved_dir = f'{cfg.model_path}',
                                                      )
                    if iteration <= update_input_poses_until:
                        pose_model.update_input_pose_with_learned()
                        pose_model.reset_delta_pose()




            compo_all_gs_ordered_renderonce = controller.compo_all_gs_ordered_renderonce
            #/////////////////////////////////////////////////////////////////////////////////////////
            # save some of the trn imgs
            # do this before test--so that can save the doing parse_cam again
            # if monitor_trn_by_render_seperately and (iteration % 1000 == 0):
            # percent = 0.9
            percent = 0
            if model_args.monitor_trn_by_render_seperately \
                and (iteration % monitor_trn_by_render_seperately_freq == 0)\
                and iteration > int(percent*training_args.iterations)\
                    :
                # row0: gt_image, image, depth
                # row1: acc, image_obj, acc_obj
                # depth_colored, _ = visualize_depth_numpy(depth.detach().cpu().numpy().squeeze(0))
                # depth_colored = depth_colored[..., [2, 1, 0]] / 255.
                # depth_colored = torch.from_numpy(depth_colored).permute(2, 0, 1).float().cuda()
                # row0 = torch.cat([gt_image, image, depth_colored], dim=2)
                gt_depth_vis = gt_depth.repeat(3, 1, 1).to(gt_image.device) / 255.0
                row0 = torch.cat([gt_image, gt_depth_vis], dim=2)
                # acc = acc.repeat(3, 1, 1)
                with torch.no_grad():
                    bg_color = [1, 1, 1] if cfg.data.white_background else [0, 0, 0]
                    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

                    image_to_show_list = [row0]
                    # if controller.obj_tool:
                    
                    # controller.set_visibility(include_list=compo_all_gs_ordered_renderonce) # extend                    
                    # assert 0,controller.include_list
                    # assert 0,controller.model_name_id.keys()
                    # include_list_tmp = controller.include_list
                    for model_name in controller.model_name_id.keys():
                        # controller.set_visibility(include_list=compo_all_gs_ordered_renderonce)# extend debug
                        # controller.set_visibility(include_list=[model_name])
                        if renderOnce and (model_name not in compo_all_gs_ordered_renderonce):
                            continue
                        # controller.set_visibility(include_list=[model_name])
                        if controller.get_visibility(model_name=model_name) or model_name == 'tissue':# or model_name == 'obj_tool1':
                            sub_gs_model = getattr(controller, model_name)
                            try:
                                assert 'tissue' in model_name
                                render_pkg,_,_= ttgs_render(viewpoint_cam, sub_gs_model, cfg.render, background,
                                                            single_compo_or_list='tissue',
                                                            tool_parse_cam_again = False,#no need for tissue 
                                                            vis_img_debug = False,
                                                            # vis_img_debug = cfg.render.dbg_vis_render,
                                                            vis_img_debug_title_more = 'again_log',
                                                            )
                            except:
                                assert 'obj_tool' in model_name
                                # render_which = 'obj_tool1'
                                # render_which = controller.candidate_model_names['obj_model_cand'][-1]
                                render_which = model_name
                                render_pkg,_,_= ttgs_render(viewpoint_cam, sub_gs_model, cfg.render, background,
                                                        debug_getxyz_ttgs = debug_getxyz_ttgs,
                                                        ttgs_model = controller,
                                                        # single_compo_or_list='tool',
                                                        single_compo_or_list=render_which,
                                                        tool_parse_cam_again = False,#no need again 
                                                        vis_img_debug = False,
                                                        # vis_img_debug = cfg.render.dbg_vis_render,
                                                        vis_img_debug_title_more = 'again_log',

                                                        )
                            image_obj, depth_obj = render_pkg["render"], render_pkg['depth']

                            depth_obj = depth_obj.repeat(3, 1, 1).to(image_obj.device) / 255.0
                            place_holder = torch.zeros_like(depth_obj).to(depth_obj.device)
                            # row_i = torch.cat([image_obj, depth_obj, place_holder], dim=2)
                            row_i = torch.cat([image_obj, depth_obj], dim=2)
                            image_to_show_list.append(row_i)
                        # else:
                            # pass
                            # assert 0, f'{model_name} {controller.model_name_id.keys()}'
                            # assert model_name == 'tissue' , f'{model_name} {controller.model_name_id.keys()}'

                            

                # image_to_show = torch.cat([row0, row1], dim=1)
                image_to_show = torch.cat(image_to_show_list, dim=1)
                image_to_show = torch.clamp(image_to_show, 0.0, 1.0)
                os.makedirs(f"{cfg.model_path}/log_images", exist_ok = True)
                log_img_name = f'it{iteration}_name{viewpoint_cam.image_name}_id{viewpoint_cam.id}_time{viewpoint_cam.time}'
                # log_img_name = f'id{viewpoint_cam.id}_it{iteration}_name{viewpoint_cam.image_name}_time{viewpoint_cam.time:.2f}'
                save_img_torch(image_to_show, f"{cfg.model_path}/log_images/{log_img_name}.jpg")
            #/////////////////////////////////////////////////////////////////////////////////////////
            # log psnr for test

            with torch.no_grad():
                if do_test and iteration % test_freq == 0:
                    #render first
                    test_viewpoint_stack = scene.getTestCameras().copy()
                    test_viewpoint_cam: Camera = test_viewpoint_stack.pop(randint(0, len(test_viewpoint_stack) - 1))

                    test_render_pkg_all,test_compo_all_gs_ordered_idx,  test_Ll1, test_loss, test_loss_dict,\
                        test_gt_image,test_gt_depth,test_image_all,test_tissue_mask,test_tool_mask,test_tool_masks_dict,\
                            test_radii_all_compo_adc,test_visibility_filters_all_compo_adc,test_model_names_all_compo_adc,\
                                test_viewspace_point_tensors_all_compo_adcdict,test_visibility_filters_all_compo_adc,\
                                    test_model_names_all_compo_adc = render_ttgs_n_compute_loss(controller,test_viewpoint_cam,cfg,training_args,optim_args,
                                            renderOnce = True,
                                            debug_getxyz_ttgs=debug_getxyz_ttgs,
                                            iteration = iteration,
                                            skip_loss_compute=True,
                                            vis_img_debug_title_more='test',
                                            tool_parse_cam_again_renderonce = True,

                                        )

                    test_renderSeperate_image_dict = {}
 
                    if compute_more_metrics_flag:
                        ema_psnr_for_log_tissue_test,ema_psnr_for_log_tool_test,more_to_log = \
                            compute_more_metrics(test_gt_image,
                                                renderOnce = True,
                                                image_all = test_image_all,
                                                cfg = cfg,
                                                tissue_mask=test_tissue_mask,
                                                tool_mask=test_tool_mask,
                                                more_to_log=more_to_log,
                                                use_ema=use_ema_test,
                                                ema_psnr_for_log_tissue=ema_psnr_for_log_tissue_test,
                                                ema_psnr_for_log_tool=ema_psnr_for_log_tool_test,
                                                dir_append = '_test',
                                                renderSeperate_image_dict = test_renderSeperate_image_dict,
                                                tool_masks_dict = test_tool_masks_dict,
                                                detail_log_tool = True,

                                                )
                    else:
                        more_to_log=more_to_log
                    




        with torch.no_grad():
            if iteration < optim_args.densify_until_iter :
                assert optim_args.densify_until_iter_tool <= optim_args.densify_until_iter
                
                controller.set_visibility(include_list=list(set(controller.model_name_id.keys()) ))
                controller.set_max_radii2D_all_models(radiis = radii_all_compo_adc, 
                                                      visibility_filters = visibility_filters_all_compo_adc,
                                                      model_names = model_names_all_compo_adc)
                controller.add_densification_stats_all_models(viewspace_point_tensors = viewspace_point_tensors_all_compo_adcdict if not renderOnce else render_pkg_all['viewspace_points'], 
                                                              visibility_filters = visibility_filters_all_compo_adc,
                                                              model_names = model_names_all_compo_adc,
                                                              compo_all_gs_ordered_idx = compo_all_gs_ordered_idx if renderOnce else None,
                                                              )

                opacity_threshold = optim_args.opacity_threshold_fine_init - iteration*(optim_args.opacity_threshold_fine_init - optim_args.opacity_threshold_fine_after)/(optim_args.densify_until_iter)  
                densify_threshold = optim_args.densify_grad_threshold_fine_init - iteration*(optim_args.densify_grad_threshold_fine_init - optim_args.densify_grad_threshold_after)/(optim_args.densify_until_iter )  

                if renderOnce:
                    try:
                        getattr(controller, 'viewpoint_camera')
                    except:
                        assert compo_all_gs_ordered_renderonce == ['tissue'], compo_all_gs_ordered_renderonce
                        controller.viewpoint_camera = viewpoint_cam

                # densify and prune
                if iteration > optim_args.densify_from_iter and iteration % optim_args.densification_interval == 0 :
                    size_threshold = 20 if iteration > optim_args.opacity_reset_interval else None
                    controller.densify_and_prune(max_grad = densify_threshold, 
                                                    min_opacity = opacity_threshold, 
                                                exclude_list = [],
                                                extent = scene.cameras_extent, 
                                                max_screen_size = size_threshold,
                                                skip_prune = True,
                                                current_iter = iteration,
                                                densify_until_iter_tool = optim_args.densify_until_iter_tool,

                                                )
                    
                if iteration > optim_args.pruning_from_iter and iteration % optim_args.pruning_interval == 0:
                    size_threshold = 40 if iteration > optim_args.opacity_reset_interval else None
                    controller.densify_and_prune(max_grad = densify_threshold, 
                                                    min_opacity = opacity_threshold, 
                                                exclude_list = [],
                                                extent = scene.cameras_extent, 
                                                max_screen_size = size_threshold,
                                                skip_densify = True,
                                                current_iter = iteration,
                                                max_iter = training_args.iterations,
                                                densify_until_iter_tool = optim_args.densify_until_iter_tool,

                                                
                                                )
                if iteration % optim_args.opacity_reset_interval == 0:
                    print("reset opacity")
                    controller.reset_opacity()
            
            if iteration < training_args.iterations:
                controller.update_optimizer()

        with torch.no_grad():
            more_to_log={**more_to_log,**loss_dict}
            if render_stree_param_for_ori_train_report!= None and iteration%tb_report_freq==0:
                training_report(tb_writer, iteration, Ll1, loss, iter_start.elapsed_time(iter_end), 
                            training_args.test_iterations, 
                            scene, 
                            more_to_log = more_to_log,
                            log_in_pts_num= not disable_log_in_pts_num,
                            )

            if iteration < training_args.iterations:
                controller.update_optimizer()
            if (iteration in training_args.checkpoint_iterations):
                pass

 


from gaussian_renderer.ttgs_renderer import TTGaussianRenderer
def training_report_ttgs(tb_writer, 
                          iteration, 
                          scalar_stats, 
                          tensor_stats, 
                          testing_iterations, 
                          scene: Scene,
                          renderer: TTGaussianRenderer,
                          cfg = None,
                          ):
    if tb_writer:
        try:
            for key, value in scalar_stats.items():
                tb_writer.add_scalar('train/' + key, value, iteration)
            for key, value in tensor_stats.items():
                tb_writer.add_histogram('train/' + key, value, iteration)
        except:
            print('Failed to write to tensorboard')
            
            
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test/test_view', 
                               'cameras' : scene.getTestCameras()},
                              {'name': 'test/train_view', 
                               'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderer.render(viewpoint, scene.gaussians_or_controller)["rgb"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    # try to get all the masks
                    if hasattr(viewpoint, 'tissue_mask'):
                        tissue_mask = viewpoint.tissue_mask.cuda().bool()
                    else:
                        tissue_mask = torch.ones_like(gt_image[0]).bool()
                    
                    if cfg.model.nsg.include_tissue:
                        from utils.loss_utils import l1_loss
                        l1_test += l1_loss(image, gt_image, tissue_mask).mean().double()
                        psnr_test += psnr(image, gt_image, tissue_mask).mean().double()
                    else:
                        assert 0,'always include tissue'


                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("test/opacity_histogram", scene.gaussians_or_controller.get_opacity, iteration)
            tb_writer.add_scalar('test/points_total', scene.gaussians_or_controller.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()