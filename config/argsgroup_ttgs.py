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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup_stree:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)
        
    def extract(self, args, \
                find_redundant = True, \
                remain_redundant = False,
                    dbg = False):

        resetted = {}
        missing = {}
        redundant = {}

        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
                resetted[arg[0]] = arg[1]
            else:
                missing[arg[0]] = arg[1]
        #////////////////
        if find_redundant:
            for var in vars(self):
                if (var not in vars(args).keys()) and (var not in [ "_"+arg for arg in vars(args).keys()]):
                    redundant[var] = vars(self)[var]
                    if remain_redundant:
                        setattr(group, var, vars(self)[var])

        if not dbg:
            return [group]
        else:
            print('////////////////////////////////')
            print('Reset')
            print(resetted)
            print('Redundant')
            print(redundant)
            return group,missing,resetted,redundant
        
class EvalParams(ParamGroup_stree):
    def __init__(self, parser):
        self.skip_train = False 
        self.skip_test = False 
        self.eval_train = False
        self.eval_test = True
        super().__init__(parser, "Eval Parameters")

class TrainParams(ParamGroup_stree):
    def __init__(self, parser):
        # self = CN()
        self.debug_from = -1
        self.detect_anomaly = False
        self.test_iterations = [7000, 30000]
        self.save_iterations = [7000, 30000]
        self.iterations = 30000
        self.quiet = False
        self.checkpoint_iterations = [30000]
        self.start_checkpoint = None
        self.importance_sampling = False
        super().__init__(parser, "Train Parameters")

class OptParams(ParamGroup_stree):
    def __init__(self, parser):
        #fdm
        self.position_lr_init = 0.00016 # position_lr_init_{bkgd, obj ...}, similar to the following
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30000

        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        # densification and pruning
        self.percent_dense = 0.01 
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15000
        self.densify_until_iter_tool = 800
        self.densify_grad_threshold = 0.0002 # densify_grad_threshold_{bkgd, obj ...}
        self.densify_grad_abs_bkgd = False # densification strategy from AbsGS
        self.densify_grad_abs_obj = False 
        self.max_screen_size = 20
        self.min_opacity = 0.005
        self.percent_big_ws = 0.1
        # tool densify
        self.tool_percent_dense = 0.0


        # loss weight
        self.lambda_l1 = 1.
        self.lambda_dssim = 0.2
        self.lambda_sky = 0.
        self.lambda_sky_scale = []
        self.lambda_semantic = 0.
        self.lambda_reg = 0.
        self.lambda_depth_lidar = 0.
        self.lambda_depth_mono = 0.
        self.lambda_normal_mono = 0.
        self.lambda_color_correction = 0.
        self.lambda_pose_correction = 0.
        self.lambda_scale_flatten = 0.
        self.lambda_opacity_sparse = 0.
        
        # extend fdm model needed
        self.deformation_lr_init = 0.00016
        self.deformation_lr_final = 0.000016
        self.deformation_lr_delay_mult = 0.01
        self.opacity_threshold_coarse = 0.005
        self.opacity_threshold_fine_init = 0.005
        self.opacity_threshold_fine_after = 0.005
        self.densify_grad_threshold_coarse = 0.0002
        self.densify_grad_threshold_fine_init = 0.0002
        self.densify_grad_threshold_after = 0.0002
        self.pruning_from_iter = 500
        self.pruning_interval = 100

        # extend posemodel needed

        # self.tool_adc_mode = 'init'
        self.tool_adc_mode = ''
        self.do_current_adc_since_iter = 2500

        self.track_warmup_steps = 0
        self.tool_fdm_warmup_steps = 0

        self.track_position_lr_delay_mult = 0.01
        self.track_position_lr_init =  0.005
        self.track_position_lr_final = 5.0e-5
        self.track_position_max_steps = 30000

        self.track_rotation_lr_delay_mult = 0.01
        self.track_rotation_lr_init = 0.001
        self.track_rotation_lr_final = 1.0e-5
        self.track_rotation_max_steps = 30000

        #tool
        self.tool_prune_big_points = True
        self.obj_pose_init = '0'
        self.obj_pose_rot_optim_space = 'rpy' #'lie'
        self._disable_tb = False
        self._add_debug = False
        self._method = 'ttgs'#'deform3dgs'


        super().__init__(parser, "Optim Parameters")

class ModParams(ParamGroup_stree):
    def __init__(self, parser):
        from config.yacs import CfgNode as CN
        # self = CN()
        self.gaussian = CN()
        self.gaussian.sh_degree = 3
        self.gaussian.fourier_dim = 1 # fourier spherical harmonics dimension
        self.gaussian.fourier_scale = 1.
        self.gaussian.flip_prob = 0. # symmetry prior for rigid objects, flip gaussians with this probability during training
        self.gaussian.semantic_mode = 'logits'

        self.nsg = CN()
        self.nsg.include_bkgd = False # include background
        self.nsg.include_sky = False # include sky cubemap
        self.nsg.include_tissue = True  #True # include background
        self.nsg.include_obj = True # include object
        self.nsg.include_obj_pose = True #True # include object

        self.sky = CN()
        self.sky.resolution = 1024
        self.sky.white_background = True

        #### Note: We have not fully tested this.
        self.use_color_correction = False # If set to True, learn transformation matrixs for appearance embedding
        self.color_correction = CN() 
        self.color_correction.mode = 'image' # If set to 'image', learn separate embedding for each image. If set to 'sensor', learn a single embedding for all images captured by one camera senosor. 
        self.color_correction.use_mlp = False # If set to True, regress embedding from extrinsic by a mlp. Otherwise, define the embedding explicitly.
        self.color_correction.use_sky = False # If set to True, using spparate embedding for background and sky
        # Alternative choice from GOF: https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/scene/appearance_network.py
        self.use_pose_correction = False # If set to True, use pose correction for camera poses. 
        self.pose_correction = CN()
        self.pose_correction.mode = 'image' # If set to 'image', learn separate correction matrix for each image. If set to 'frame', learn a single correction matrix for all images corresponding to the same frame timestamp. 
        ####

        self.fdm = CN()
        self.fdm.net_width = 64
        self.fdm.timebase_pe = 4
        self.fdm.defor_depth = 1
        self.fdm.posebase_pe = 10
        self.fdm.scale_rotation_pe = 2
        self.fdm.opacity_pe = 2
        self.fdm.timenet_width = 64
        self.fdm.timenet_output = 32
        self.fdm.bounds = 1.6  

        self.fdm.ch_num = 10
        self.fdm.curve_num = 17
        self.fdm.init_param = 0.01

        self.tool_mask = None
        self.tool_init_mode = None
        self.inited_pcd_noise_removal = False
        self.supervise_depth_noise_ignore_tgt = []
        self.tissue_init_mode = None
        self.model_path = None
        self.source_path = None
        self.extra_mark = None

        self.camera_extent = None
        self.load_cotrackerPnpPose = False
        self.init_detail_params_dict = {'test':1}


        # extend extend: ttgs only
        self.renderOnce = True
        self.compo_all_gs_ordered_renderonce = []
        self.remain_redundant_default_param = True
        self.monitor_trn_by_render_seperately = True
        self.use_opt_track = True
        self.use_fdm_tool = False
        self.disable_tool_fdm_scale = False
        #////////////////////////////
        self.basis_type = 'gaussian'
        self.poly_degree = 2
        #//////////////////////////////

        # extend extend: shared
        self.do_test = False
        self.tb_report_freq = 20
        self.test_freq = 20
        self.use_ema_train = False
        self.use_ema_test = False
        self.dbg_print = False
        self.dbg_vis_adc = False
        self.tool_mask_loss_src = []
        self.tissue_mask_loss_src = ['depth','color']
        self.reg_fg_tool = False
        self.reg_bg_tissue = False
        self.do_asap_for_tools = False


        self.process_tissue_mask_trn = None
        self.process_tool_mask_trn = None 
        self.process_tissue_mask_init = None 
        self.process_tool_mask_init = None 

        super().__init__(parser, "Model Parameters")
 
class DataParams(ParamGroup_stree):
    def __init__(self, parser):
        self.white_background = False # If set to True, use white background. Should be False when using sky cubemap.
        self.use_colmap_pose = False # If set to True, use colmap to recalibrate camera poses as input (rigid bundle adjustment now).
        self.filter_colmap = False # If set to True, filter out SfM points by camera poses.
        self.box_scale = 1.0 # Scale the bounding box by this factor.
        self.split_test = -1 
        self.shuffle = True
        self.eval = True
        self.type = 'Colmap'
        self.images = 'images'
        self.use_semantic = False
        self.use_mono_depth = False
        self.use_mono_normal = False
        self.use_colmap = True
        
        super().__init__(parser, "Data Parameters")

class RenderParams(ParamGroup_stree):
    def __init__(self, parser):
        # self = CN()
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.scaling_modifier = 1.0
        self.fps = 24
        self.render_normal = False
        self.save_video = True
        self.save_image = True
        self.coord = 'world' # ['world', 'vehicle']
        self.concat_cameras = []

        # extend extend in pipe
        self.dbg_vis_render = False


        super().__init__(parser, "Render Parameters")

class ViewerParams(ParamGroup_stree):
    def __init__(self, parser):
        # self = CN()
        self.frame_id = 0 # Select the frame_id (start from 0) to save for viewer

        super().__init__(parser, "Viewer Parameters")

# the pipe args group etc
OTHER_PARAM_DICT  = {
    "workspace": os.environ['PWD'],
    "loaded_iter": -1,
    "ip": "127.0.0.1",
    "port": 6009,
    "data_device": "cuda",
    "mode": "train",
    "task": "hello",  # task folder name
    "exp_name": "test",  # experiment folder name
    "gpus": [0],  # list of gpus to use
    "debug": False,
    "resume": True,  # If set to True, resume training from the last checkpoint
    
    "source_path": "",
    "model_path": "",
    "record_dir": None,
    "resolution": -1,
    "resolution_scales": [1],

}

