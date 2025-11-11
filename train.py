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
from utils.loss_utils import l1_loss
from gaussian_renderer import ttgs_render

import sys
from scene import  Scene
from scene.flexible_deform_model import TissueGaussianModel
from scene.tool_model import ToolModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from arguments import FDMHiddenParams as ModelHiddenParams
from utils.timer import Timer
import torch.nn.functional as F

from utils.scene_utils import render_training_image
from utils.loss_utils import ssim
from metrics import cal_lpips



to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def compute_more_metrics_for_all_cams(image_tensor,gt_image_tensor,
                                        tissue_mask_tensor,
                                        tool_mask_tensor,
                                        dataset,
                                        more_to_log,
                                        use_ema = False,
                                        ema_psnr_for_log_tissue = None,
                                        ema_psnr_for_log_tool = None,
                                        dir_append = '',
                                        compute_ssim = False,
                                        compute_lpips = False,
                                        tool_masks_dict = {},
                                        detail_log_tool = False,
                            ):
    # Log PSNR in tb
    psnr_weight_ori = 0.6 if use_ema else 0# set to 0,then it would be compariable to the one computed in deform3dgs
    psnr_weight_current = 1-psnr_weight_ori
    log_psnr_name = 'ema_psnr' if use_ema else 'crt_psnr'
    log_ssim_name = 'ssim'
    log_lpips_name = 'lpips'

    ema_psnr_for_log_tissue = psnr_weight_current * psnr(image_tensor, gt_image_tensor, 
                                                tissue_mask_tensor).mean().double()
    + psnr_weight_ori * ema_psnr_for_log_tissue
    more_to_log[f'tissue/{log_psnr_name}{dir_append}'] = ema_psnr_for_log_tissue
    if compute_ssim:
        ssim_for_log_tissue = ssim(image_tensor.to(torch.double), gt_image_tensor.to(torch.double), mask = tissue_mask_tensor).mean().float() 
        more_to_log[f'tissue/{log_ssim_name}{dir_append}'] = ssim_for_log_tissue
    if compute_lpips:
        lpips_for_log_tissue = cal_lpips((image_tensor*tissue_mask_tensor).to(torch.float32), 
                                        (gt_image_tensor*tissue_mask_tensor).to(torch.float32), 
                                        ).mean().float() 
        more_to_log[f'tissue/{log_lpips_name}{dir_append}'] = lpips_for_log_tissue

    for tool_name, tool_mask_tensor in tool_masks_dict.items():

        if not detail_log_tool:
            if tool_name != 'tool':
                continue


        ema_psnr_for_log_tool = psnr_weight_current * psnr(image_tensor, gt_image_tensor, 
                                                    tool_mask_tensor).mean().double()
        + psnr_weight_ori * ema_psnr_for_log_tool
        more_to_log[f'{tool_name}/{log_psnr_name}{dir_append}'] = ema_psnr_for_log_tool
        if compute_ssim:
            ssim_for_log_tool = ssim(image_tensor.to(torch.double), gt_image_tensor.to(torch.double), mask = tool_mask_tensor).mean().float()
            more_to_log[f'{tool_name}/{log_ssim_name}{dir_append}'] = ssim_for_log_tool
        if compute_lpips:
            lpips_for_log_tool = cal_lpips((image_tensor*tool_mask_tensor).to(torch.float32), 
                                            (gt_image_tensor*tool_mask_tensor).to(torch.float32), 
                                            ).mean().float()
            more_to_log[f'{tool_name}/{log_lpips_name}{dir_append}'] = lpips_for_log_tool

    return  more_to_log,ema_psnr_for_log_tissue,ema_psnr_for_log_tool

def render_viewpoint_cams(viewpoint_cams,gaussians, pipe, background,
                          dbg_vis_render = False,
                          vis_img_debug_title_more = 'trn',

                          ):
    '''
    use for training
    '''
    images = []
    depths = []
    gt_images = []
    gt_depths = []
    masks = []
    masks_tissue_dbg = []
    
    radii_list = []
    visibility_filter_list = []
    viewspace_point_tensor_list = []
    
    for viewpoint_cam in viewpoint_cams:
        # also singel frame
        assert len(viewpoint_cams)==1
        viewpoint_cam = viewpoint_cams[0]
        render_pkg,_,_ = ttgs_render(viewpoint_cam, gaussians, pipe, background,
                            single_compo_or_list='tissue',
                            vis_img_debug = dbg_vis_render,
                            vis_img_debug_title_more = vis_img_debug_title_more,
                            )
        image, depth, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = viewpoint_cam.original_image.cuda().float()
        gt_depth = viewpoint_cam.original_depth.cuda().float()
        mask = viewpoint_cam.mask.cuda()
        mask_tissue_dbg = viewpoint_cam.raw_tissue.cuda()
        images.append(image.unsqueeze(0))
        depths.append(depth.unsqueeze(0))
        gt_images.append(gt_image.unsqueeze(0))
        gt_depths.append(gt_depth.unsqueeze(0))
        masks.append(mask.unsqueeze(0))
        masks_tissue_dbg.append(mask_tissue_dbg.unsqueeze(0))
        radii_list.append(radii.unsqueeze(0))
        visibility_filter_list.append(visibility_filter.unsqueeze(0))
        viewspace_point_tensor_list.append(viewspace_point_tensor)

    radii = torch.cat(radii_list,0).max(dim=0).values
    visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
    image_tensor = torch.cat(images,0)
    depth_tensor = torch.cat(depths, 0)
    gt_image_tensor = torch.cat(gt_images,0)
    gt_depth_tensor = torch.cat(gt_depths, 0)
    mask_tensor = torch.cat(masks, 0)
    mask_tissue_dbg_tensor = torch.cat(masks_tissue_dbg, 0)

    return image_tensor,depth_tensor,gt_image_tensor,gt_depth_tensor,mask_tensor,mask_tissue_dbg_tensor,\
    viewspace_point_tensor_list, radii,visibility_filter

def get_tool_masks_dict(loaded_obj_names,viewpoint_cam,process_tool_mask_trn):
    tool_masks_dict = {}
    for render_which in loaded_obj_names:
        tool_mask_obj_i = getattr(viewpoint_cam, f'raw_{render_which}').cuda().bool()
        tool_mask_obj_i = erode_mask_torch(masks = tool_mask_obj_i,kernel_size = 10) if process_tool_mask_trn  == 'erode' else tool_mask_obj_i
        tool_masks_dict[render_which] = tool_mask_obj_i
    tool_mask = torch.stack(list(tool_masks_dict.values())).any(dim=0)
    tool_masks_dict['tool'] = tool_mask
    return tool_masks_dict

def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, tb_writer, train_iter, timer,
                         ):
    
    do_test = dataset.do_test
    tb_report_freq = dataset.tb_report_freq
    test_freq = dataset.test_freq
    assert tb_report_freq == test_freq ,f'avoid waste testing'

    use_ema_test=dataset.use_ema_test
    use_ema_train=dataset.use_ema_train

    first_iter = 0
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None

    
    final_iter = train_iter
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras()
        
    ema_psnr_for_log_tissue_trn = 0.0
    ema_psnr_for_log_tool_trn = 0.0
    ema_psnr_for_log_tissue_test = 0.0
    ema_psnr_for_log_tool_test = 0.0


    for iteration in range(first_iter, final_iter+1):        
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()
        idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cams = [viewpoint_stack[idx]]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # render 
        assert len(viewpoint_cams)==1
        image_tensor,depth_tensor,gt_image_tensor,gt_depth_tensor,mask_tensor,mask_tissue_dbg_tensor,\
            viewspace_point_tensor_list,radii, visibility_filter = render_viewpoint_cams(viewpoint_cams,gaussians, pipe, background,
                                                                                         dbg_vis_render = pipe.dbg_vis_render,
                                                                                         vis_img_debug_title_more='trn',
                                                                                         )

        from utils.general_utils import erode_mask_torch
        if dataset.process_tissue_mask_trn == 'erode':
            assert dataset.tool_mask == 'use'
            trn_mask_tensor = erode_mask_torch(masks = mask_tensor.squeeze(0),kernel_size = 50)
        else:
            trn_mask_tensor = mask_tensor
        Ll1_tissue = torch.tensor(0.).cuda()
        depth_loss_tissue = torch.tensor(0.).cuda()
        if 'color' in dataset.tissue_mask_loss_src:
            assert dataset.process_tissue_mask_trn in [None,'erode']
            Ll1_tissue = l1_loss(image_tensor, gt_image_tensor, 
                                 mask = trn_mask_tensor,
                                 )

        if 'depth' in dataset.tissue_mask_loss_src:
            if (gt_depth_tensor!=0).sum() >= 10:
                depth_tensor[depth_tensor!=0] = 1 / depth_tensor[depth_tensor!=0]
                gt_depth_tensor[gt_depth_tensor!=0] = 1 / gt_depth_tensor[gt_depth_tensor!=0]
        
                depth_loss_tissue = l1_loss(depth_tensor, gt_depth_tensor, 
                                            mask = trn_mask_tensor,
                                            )
        Ll1 = Ll1_tissue
        depth_loss = depth_loss_tissue
        loss = Ll1 + depth_loss 
        loss.backward()

        loss_dict = {}
        loss_dict['loss/Tissue_loss'] = Ll1_tissue.item()+depth_loss_tissue.item()
        loss_dict['loss/Color_loss'] = Ll1.item()
        loss_dict['loss/Depth_loss'] = depth_loss.item()
        loss_dict['loss/Total_loss'] = loss.item()

        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor_list[-1])
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()

        more_to_log = {}
        with torch.no_grad():
            assert isinstance(scene.gaussians_or_controller, TissueGaussianModel)
            assert dataset.process_tissue_mask_trn in [None,'erode']

            tool_masks_dict = get_tool_masks_dict(loaded_obj_names = scene.loaded_obj_names,viewpoint_cam = viewpoint_cams[0],
                                                  process_tool_mask_trn = dataset.process_tool_mask_trn)

            more_to_log,ema_psnr_for_log_tissue_trn,ema_psnr_for_log_tool_trn = compute_more_metrics_for_all_cams(image_tensor,gt_image_tensor,
                                                tissue_mask_tensor = trn_mask_tensor if dataset.process_tissue_mask_trn == 'erode' else mask_tissue_dbg_tensor,
                                                tool_mask_tensor=~mask_tissue_dbg_tensor,
                                                tool_masks_dict = tool_masks_dict,
                                                dataset = dataset,
                                                more_to_log=more_to_log,
                                                use_ema = use_ema_train,
                                                ema_psnr_for_log_tissue = ema_psnr_for_log_tissue_trn,
                                                ema_psnr_for_log_tool = ema_psnr_for_log_tool_trn,
                                                dir_append = '_trn',
                                                detail_log_tool = True,
                                                )
        
        with torch.no_grad():
            if do_test and iteration % test_freq == 0:
                from scene.cameras import Camera
                test_viewpoint_stack = scene.getTestCameras().copy()
                test_viewpoint_cam: Camera = test_viewpoint_stack.pop(randint(0, len(test_viewpoint_stack) - 1))
                test_viewpoint_cams = [test_viewpoint_cam]
                
                assert len(test_viewpoint_cams)==1
                test_image_tensor,test_depth_tensor,test_gt_image_tensor,test_gt_depth_tensor,test_mask_tensor,test_mask_tissue_dbg_tensor,\
                    test_viewspace_point_tensor_list,test_radii,test_visibility_filter = render_viewpoint_cams(test_viewpoint_cams,gaussians, pipe, background,
                                                                                                               dbg_vis_render = pipe.dbg_vis_render,
                                                                                                               vis_img_debug_title_more='test',
                                                                                                               )



                tool_masks_dict = get_tool_masks_dict(loaded_obj_names = scene.loaded_obj_names,viewpoint_cam = test_viewpoint_cams[0],\
                                                      process_tool_mask_trn = dataset.process_tool_mask_trn)


                assert dataset.process_tissue_mask_trn in [None,'erode']
                more_to_log,ema_psnr_for_log_tissue_test,ema_psnr_for_log_tool_test = compute_more_metrics_for_all_cams(test_image_tensor,test_gt_image_tensor,
                                                    tissue_mask_tensor = test_mask_tissue_dbg_tensor if dataset.process_tissue_mask_trn != 'erode' else erode_mask_torch(masks = test_mask_tissue_dbg_tensor.squeeze(0),kernel_size = 50),
                                                    tool_mask_tensor=~test_mask_tissue_dbg_tensor,
                                                    dataset=dataset,
                                                    more_to_log=more_to_log,
                                                    use_ema = use_ema_test,
                                                    ema_psnr_for_log_tissue = ema_psnr_for_log_tissue_test,
                                                    ema_psnr_for_log_tool = ema_psnr_for_log_tool_test,
                                                    dir_append = '_test',
                                                    tool_masks_dict=tool_masks_dict,
                                                    detail_log_tool = True,

                                                    )

        # Progress bar
        total_point = gaussians._xyz.shape[0]
        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}",
                                        f"psnr_tissue": f"{ema_psnr_for_log_tissue_trn:.{2}f}",
                                        f"psnr_tool": f"{ema_psnr_for_log_tool_trn:.{2}f}",
                                        "point":f"{total_point}"})
            progress_bar.update(10)
        if iteration == opt.iterations:
            progress_bar.close()

        with torch.no_grad():
            if iteration % tb_report_freq == 0:
                training_report(tb_writer, iteration, Ll1, loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, 
                                more_to_log={**more_to_log,**loss_dict},
                                log_in_pts_num=True,
                                )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, 'fine')

        with torch.no_grad():
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)
                opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  

                # densify and prune
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    gaussians.densify_and_prune(max_grad = densify_threshold, 
                                                min_opacity = opacity_threshold, 
                                                extent = scene.cameras_extent, 
                                                max_screen_size = size_threshold,
                                                skip_prune = True)
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 40 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(max_grad = densify_threshold, 
                                                min_opacity = opacity_threshold, 
                                                extent = scene.cameras_extent, 
                                                max_screen_size = size_threshold,
                                                skip_densify = True)
                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, extra_mark,args = None,
             ):


    assert expname == args.model_path, f'{expname} {args.model_path}'

    tb_writer = prepare_output_and_logger(model_path=expname, write_args=args)
    gaussians = TissueGaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset)
    scene.gs_init(gaussians_or_controller=gaussians,
                  reset_camera_extent=dataset.camera_extent)
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, tb_writer, opt.iterations,timer,
                         )

def prepare_output_and_logger(model_path,write_args = None):  
    if not model_path:
        assert 0, model_path
    print("Output folder: {}".format(model_path))
    


    os.makedirs(model_path, exist_ok = True)
    with open(os.path.join(model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(write_args))))
    tb_writer = None
    
    if write_args.disable_tb:
        return None


    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, elapsed, testing_iterations, scene : Scene, 
                    more_to_log = {},
                    log_in_pts_num = False,
                    ):
    
    if tb_writer:
        tb_writer.add_scalar(f'other/iter_time', elapsed, iteration)

        if log_in_pts_num:
            from scene.tt_gaussian_model import TTGaussianModel
            if isinstance(scene.gaussians_or_controller, TTGaussianModel):
                for obj_name in  scene.gaussians_or_controller.candidate_model_names['obj_model_cand']+ scene.gaussians_or_controller.candidate_model_names['tissue_model']:
                    tb_writer.add_scalar(f'other/total_points_{obj_name}', getattr(scene.gaussians_or_controller,f'{obj_name}').get_xyz.shape[0], iteration)

                tb_writer.add_scalar('other/total_points_all', scene.gaussians_or_controller.tissue.get_xyz.shape[0]
                                     +scene.gaussians_or_controller.obj_tool1.get_xyz.shape[0], iteration)
            else:
                if isinstance(scene.gaussians_or_controller, TissueGaussianModel):
                    tgt = 'tissue'
                elif isinstance(scene.gaussians_or_controller, ToolModel):
                    tgt = 'tool'
                else:
                    assert 0,scene.gaussians_or_controller
                tb_writer.add_scalar(f'other/total_points_{tgt}', scene.gaussians_or_controller.get_xyz.shape[0], iteration)
        
        if more_to_log != {}:
            for k,v in more_to_log.items():
                tb_writer.add_scalar(f'{k}', v, iteration)



def training_report_v0(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, 
                    ):
    
    if tb_writer:
        tb_writer.add_scalar(f'train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'iter_time', elapsed, iteration)

        from scene.tt_gaussian_model import TTGaussianModel
        if isinstance(scene.gaussians_or_controller, TTGaussianModel):
            tb_writer.add_scalar('total_points', scene.gaussians_or_controller.tissue.get_xyz.shape[0], iteration)
            tb_writer.add_scalar('total_points_tool', scene.gaussians_or_controller.obj_tool1.get_xyz.shape[0], iteration)
            tb_writer.add_scalar('total_points_all', scene.gaussians_or_controller.tissue.get_xyz.shape[0]+scene.gaussians_or_controller.obj_tool1.get_xyz.shape[0], iteration)
        else:
            tb_writer.add_scalar('total_points', scene.gaussians_or_controller.get_xyz.shape[0], iteration)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    torch.cuda.empty_cache()

    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*500 for i in range(0,120)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000,])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[1])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "endonerf/pulling_fdm")
    parser.add_argument("--configs", type=str, default = "arguments/default.py")

    args = parser.parse_args(sys.argv[1:])
    # Get list of source paths
    import copy
    source_paths = args.source_path.split(',')
    expnames = args.expname.split(',')
    print('expnames',expnames)
    print('source_paths',source_paths)
    assert len(source_paths)==len(expnames)

    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)


    print(f"Found {len(source_paths)} source paths to process: {source_paths}")
    for source_path,expname in zip(source_paths,expnames):
        current_args=copy.deepcopy(args)
        current_args.source_path = source_path
        current_args.expname = expname

        if current_args.configs:
            current_args = merge_hparams(current_args, config)
            
            assert current_args.process_tissue_mask_trn in ['erode',None]

            mask_append = ''
            init_append = ''
            optim_append = ''
            loss_append = ''
            reg_append = ''
            pose_append = ''
            tool_append = ''
            adc_append = ''
            if current_args.method == 'ttgs':
                assert current_args.tool_mask == 'use',f' for ttgs,we let tool_mask be use n get all masks'
                for loss_term in current_args.tissue_mask_loss_src:
                    assert loss_term in ['depth','color']
                for loss_term in current_args.tool_mask_loss_src:
                    assert loss_term in ['depth','color']

                pose_append += f'_corrected{int(current_args.use_opt_track)}_ini{current_args.obj_pose_init}'
                assert current_args.process_tool_mask_trn in ['erode',None]

                if current_args.reg_fg_tool:
                    reg_append += '_fgTool_with_ASAP' if current_args.do_asap_for_tools else '_fgTool'
                if current_args.reg_bg_tissue:
                    reg_append += '_bgTissue'
                reg_append = f'_Reg{reg_append}' if reg_append!='' else reg_append
                
                tool_append = '_Toolfdm' if current_args.use_fdm_tool else '_Toolrigid'
                tool_append = f'_Toolfdm{current_args.basis_type}' if current_args.use_fdm_tool else '_Toolrigid'
                if current_args.use_fdm_tool:
                    assert current_args.basis_type in ['gaussian','polynomial']
                tool_append = f'woscale' if current_args.use_fdm_tool and current_args.disable_tool_fdm_scale else tool_append

                if current_args.tool_adc_mode != '':
                    assert '2D' in current_args.tool_adc_mode or '6D' in current_args.tool_adc_mode
                    assert 'init' in current_args.tool_adc_mode or 'current' in current_args.tool_adc_mode
                    assert '6Dcurrent' not in current_args.tool_adc_mode

                adc_append = f'_{current_args.tool_adc_mode}' if current_args.tool_adc_mode != '' else adc_append
                if '2Dcurrent' in current_args.tool_adc_mode:
                    adc_append += f'{current_args.do_current_adc_since_iter}'

            elif current_args.method == 'deform3dgs':
                for loss_term in current_args.tissue_mask_loss_src:
                    assert loss_term in ['depth','color']
                if current_args.process_tool_mask_trn == 'erode':
                    assert current_args.tool_mask == 'use'

            else:
                assert 0,current_args.method
            tissue_mask_loss_src_color_depth_flag = [str(int('color' in current_args.tissue_mask_loss_src)),str(int('depth' in current_args.tissue_mask_loss_src))]
            tool_mask_loss_src_color_depth_flag = [str(int('color' in current_args.tool_mask_loss_src)),str(int('depth' in current_args.tool_mask_loss_src))]



            process_tissue_mask_trn = '' if current_args.process_tissue_mask_trn == None else current_args.process_tissue_mask_trn
            process_tool_mask_trn = '' if current_args.process_tool_mask_trn == None else current_args.process_tool_mask_trn

            if hasattr(current_args,'tool_mask'):
                mask_append += f'_{current_args.tool_mask}'
            
            process_tissue_mask_init = '' if current_args.process_tissue_mask_init == None else current_args.process_tissue_mask_init
            process_tool_mask_init = '' if current_args.process_tool_mask_init == None else current_args.process_tool_mask_init
            tissue_init_mode = current_args.tissue_init_mode
            tool_init_mode = current_args.tool_init_mode
            init_modes = [tissue_init_mode,tool_init_mode]
            for i,init_mode in enumerate(init_modes):
                init_detail = ''
                if init_mode in ['adaptedMAPF']:
                    occlu_interval = current_args.init_detail_params_dict['occlu_interval']
                    deform_interval = current_args.init_detail_params_dict['deform_interval']
                    add4occlu = current_args.init_detail_params_dict['add4occlu']
                    add4deform = current_args.init_detail_params_dict['add4deform']
                    init_detail += f'_occlu{occlu_interval}' if add4occlu else ''
                    init_detail += f'_deform{deform_interval}' if add4deform else ''
                    init_modes[i] = init_mode
            tissue_init_mode,tool_init_mode = init_modes 
            if current_args.tool_percent_dense!=0:
                tool_init_mode += f'_pct{current_args.tool_percent_dense}'
            
            if current_args.method == 'ttgs':
                if current_args.compo_all_gs_ordered_renderonce == []:
                    varient = 'ALL' 
                    print('Comprehensive Reconstruction....reconstruction_type: ALL')
                elif 'tissue' not in current_args.compo_all_gs_ordered_renderonce:
                    varient = 'ToolOnly'
                    print('Tool Only Reconstruction....reconstruction_type: ToolOnly')
                elif current_args.compo_all_gs_ordered_renderonce == ['tissue']:
                    varient = 'TissueOnly'
                    print('Tissue Only Reconstruction....reconstruction_type: TissueOnly')
                else:
                    assert 0, current_args.compo_all_gs_ordered_renderonce
            elif current_args.method == 'deform3dgs':
                if current_args.tool_mask == 'nouse':
                    varient = 'ALL'
                elif current_args.tool_mask == 'inverse':
                    varient = 'ToolOnly'
                else:
                    assert 0,current_args.tool_mask
            else:
                assert 0,current_args.method

            assert isinstance(current_args.disable_tb, bool)
            assert isinstance(current_args.add_debug, bool)
            setattr(current_args, 'expname', f'{current_args.expname}/{varient}')
        else:
            assert 0

        current_args.save_iterations.append(current_args.iterations)
        current_args.model_path = current_args.expname
        # check if the model path exists
        if os.path.exists(current_args.model_path):
            pass
            # assert 0, f'{current_args.model_path} already exists, comment if you want to overwrite it?'
            print('Overwrite at the existing model path: ', current_args.model_path)

        print("Optimizing " + current_args.model_path)
        print(f"\nProcessing source path: {source_path}  {current_args.model_path}")
        safe_state(current_args.quiet)

        torch.autograd.set_detect_anomaly(current_args.detect_anomaly)
        if current_args.method == 'deform3dgs':
            training(lp.extract(current_args), hp.extract(current_args), op.extract(current_args), pp.extract(current_args), current_args.test_iterations, \
                current_args.save_iterations, current_args.checkpoint_iterations, current_args.start_checkpoint, current_args.debug_from, current_args.expname, 
                extra_mark=current_args.extra_mark,
                args = current_args,
                )
        elif current_args.method == 'ttgs':
            assert current_args.obj_pose_init in ['0','cotrackerpnp']
            assert current_args.obj_pose_rot_optim_space in ['rpy','lie']
            if current_args.obj_pose_init in ['cotrackerpnp']:
                assert current_args.load_cotrackerPnpPose

            from train_utils_ttgs import training_ttgsmodel
            training_ttgsmodel(current_args,)
        else:
            assert 0
        print("\nTraining complete.", current_args.model_path)



        torch.cuda.empty_cache()  # Clear GPU cache
        import gc
        gc.collect()  # Clear CPU memory