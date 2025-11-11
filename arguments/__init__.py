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

class ParamGroup:
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

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        # self._white_background = False#False
        self._white_background = True#False
        self.data_device = "cuda"
        self.eval = True
        self.render_process=False
        self.extra_mark = None
        self.camera_extent = None

        #extend
        self.tool_mask = None
        # self.init_mode = None
        self.tool_init_mode = None
        self.inited_pcd_noise_removal = False
        self.supervise_depth_noise_ignore_tgt = []
        self.tissue_init_mode = None 
        self.load_cotrackerPnpPose = False
        self.init_detail_params_dict = {'test':1}


        # extend extend: ttgs only
        self.renderOnce = True
        # self.compo_all_gs_ordered_renderonce = ['tissue','obj_tool1']
        self.compo_all_gs_ordered_renderonce = []# auto all?
        self.remain_redundant_default_param = True
        self.monitor_trn_by_render_seperately = True
        self.use_opt_track = True
        self.use_fdm_tool = False
        self.disable_tool_fdm_scale = False
        self.basis_type = 'gaussian'#polynomial#gaussian
        # self.basis_type = 'polynomial'#polynomial
        self.poly_degree = 2



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
        # self.tool_model_type = 'tool'


        self.process_tissue_mask_trn = None 
        self.process_tool_mask_trn = None 
        self.process_tissue_mask_init = None 
        self.process_tool_mask_init = None 



        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        
         # extend        
        self.dbg_vis_render = False

        super().__init__(parser, "Pipeline Parameters")

        
class FDMHiddenParams(ParamGroup):
    def __init__(self, parser):
        self.net_width = 64
        self.timebase_pe = 4
        self.defor_depth = 1
        self.posebase_pe = 10
        self.scale_rotation_pe = 2
        self.opacity_pe = 2
        self.timenet_width = 64
        self.timenet_output = 32
        self.bounds = 1.6  

        self.ch_num = 10
        self.curve_num = 17
        self.init_param = 0.01
        
        # self.basis_type = 'gaussian'#polynomial
        # # self.basis_type = 'polynomial'#polynomial
        # self.poly_degree = 2

        super().__init__(parser, "FDMHiddenParams")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.dataloader=False
        self.iterations = 30_000
        
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30000

        self.deformation_lr_init = 0.00016
        self.deformation_lr_final = 0.000016
        self.deformation_lr_delay_mult = 0.01
        # self.grid_lr_init = 0.0016
        # self.grid_lr_final = 0.00016

        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.weight_constraint_init= 1
        self.weight_constraint_after = 0.2
        self.weight_decay_iteration = 5000
        self.opacity_reset_interval = 3000
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_until_iter_tool = 800
        self.densify_grad_threshold_coarse = 0.0002
        self.densify_grad_threshold_fine_init = 0.0002
        self.densify_grad_threshold_after = 0.0002
        self.pruning_from_iter = 500
        self.pruning_interval = 100
        self.opacity_threshold_coarse = 0.005
        self.opacity_threshold_fine_init = 0.005
        self.opacity_threshold_fine_after = 0.005
        
        # extend tool
        self.tool_percent_dense = 0.0


        #ttgs
        self.tool_prune_big_points = False
        self.densify_grad_threshold_obj = 0.0002
        self.percent_big_ws = 0.1
        self.obj_pose_init = '0'
        self.obj_pose_rot_optim_space = 'rpy' #'lie'

        self._disable_tb = False 
        self._add_debug = False 
        self.method = 'ttgs'#'deform3dgs'

        # extend posemodel needed
        self.tool_adc_mode = ''
        self.do_current_adc_since_iter = 2500,
        
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

        super().__init__(parser, "Optimization Parameters")

def ambigious_search_cfg(args_cmdline, allow_multiple = True):
    # Search for matching directories
    import glob
    base_dir = os.path.dirname(args_cmdline.model_path)
    search_substring = os.path.basename(args_cmdline.model_path)
    print(f"base_dir: {base_dir}")
    print(f"search_substring: {search_substring}")

    # Use glob to find directories with the substring in their name
    matching_dirs = [
        d for d in glob.glob(f"{base_dir}/*") 
        if os.path.isdir(d) and search_substring in os.path.basename(d)
    ]
    if not allow_multiple:
        assert len(matching_dirs) == 1, f'{len(matching_dirs)} matching model found? drop.  {matching_dirs} {args_cmdline.model_path}'
    else:
        assert len(matching_dirs) > 0, f'{matching_dirs} {args_cmdline.model_path}'
        # matching_dirs
        print(f'Render mutilple is on:  {matching_dirs} matching models found:', matching_dirs)

    return matching_dirs#[0]


# only used for offline rendering
def get_combined_args(parser : ArgumentParser, 
                      insist = True,
                      allow_multiple = True,
                      ):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    model_paths_list = ambigious_search_cfg(args_cmdline,
                                            allow_multiple=allow_multiple)#matching_dirs[0]

    list_of_namespace = []
    for model_path in model_paths_list:
        pass
        args_cmdline.model_path = model_path#ambigious_search_cfg(args_cmdline)#matching_dirs[0]
        try:
            cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
            print("Looking for config file in", cfgfilepath)
            with open(cfgfilepath) as cfg_file:
                # print("Config file found: {}".format(cfgfilepath))
                print("Config file found: {}".format(os.path.basename(cfgfilepath)))
                cfgfile_string = cfg_file.read()
        except TypeError:
            print("Config file not found at")
            if insist:
                assert 0, f'missing {cfgfilepath} for the exp...'
            pass
        args_cfgfile = eval(cfgfile_string)

        merged_dict = vars(args_cfgfile).copy()
        for k,v in vars(args_cmdline).items():
            if v != None:
                merged_dict[k] = v
        list_of_namespace.append(Namespace(**merged_dict))
    # return Namespace(**merged_dict)
    return list_of_namespace


def save_args(args, path):
    # import argparse
    import pickle
    with open(f'{path}', 'wb') as f:
        pickle.dump(args, f)
    print(f"Arguments saved {path}successfully. used for later rendering!")
    return path
