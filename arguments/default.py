
PipelineParams = dict(
    dbg_vis_render = False, #debug_only
    )

ModelParams = dict(
    extra_mark = 'endonerf',
    camera_extent = 10, # tool shares with tissue

    # gs init
    tissue_init_mode = 'adaptedMAPF',#adaptedMAPF
    tool_init_mode = 'TF',#'MAPF', #'skipMAPF' #rand #TF #adaptedMAPF
    
    inited_pcd_noise_removal = False, 
    supervise_depth_noise_ignore_tgt = [],

    tool_mask = 'use',#'use', #'use'(default) 'inverse' 'nouse'
    
    load_cotrackerPnpPose = True,
    #used only in adaptedMAPF mode
    init_detail_params_dict = {
        'add4occlu': True,
        'occlu_interval': 8,
        'add4deform': True,
        'deform_interval': 8,
    },

    # TTGS setting
    renderOnce = True,# False # True
    
    # Critical parameter:
    # 1) control what to render: all/toolonly/tissueonly
    # 2) create the subdir with correponding exp_appendix: ALL/ToolOnly/TissueOnly
    # example usage:
    # eg: recon tool1: ['obj_tool1']
    compo_all_gs_ordered_renderonce = [], #comprehensive recon; 
    # compo_all_gs_ordered_renderonce = ['obj_tool1','obj_tool2'],# used for pulling; P2_7_171_251; P2_8_16241_16309
    # compo_all_gs_ordered_renderonce = ['obj_tool1','obj_tool2','obj_tool3'],# used for cutting
    
    remain_redundant_default_param = True,
    monitor_trn_by_render_seperately = False, # debug_only

    use_opt_track = True,
    use_fdm_tool = True,
    # tissue and tool share setting in ModelHiddenParams, 
    # set it True to reduce the ch_num by 3 for tool modelling
    disable_tool_fdm_scale = True, 
    basis_type = 'gaussian', #polynomial
    poly_degree = 2,

    reg_fg_tool = True,
    reg_bg_tissue = False,# not used
    do_asap_for_tools = False,# not used

    # disable IO/logging for quick trn
    do_test = False,
    tb_report_freq = 3000,#1
    test_freq = 3000,#1  

    use_ema_train = False,
    use_ema_test = False,

    dbg_print = False,    
    dbg_vis_adc = False,

    tissue_mask_loss_src = ['depth','color'],
    tool_mask_loss_src = ['depth','color'],

    # not used
    process_tissue_mask_trn = None,# 'erode' 'dilate' 
    process_tool_mask_trn = None ,# 'erode' 'dilate' 
    process_tissue_mask_init = None,
    process_tool_mask_init = None ,
)

OptimizationParams = dict(

    do_current_adc_since_iter = 500,
    tool_adc_mode = '', 

    coarse_iterations = 0,
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,

    iterations = 3000,

    opacity_reset_interval = 3000,
    position_lr_max_steps = 4000,

    # TTGS: tool gs trn setting
    densification_interval = 100,
    densify_from_iter = 500,
    densify_until_iter = 800, #-1 represent no densify
    densify_until_iter_tool = 800,
    densify_grad_threshold_coarse = 0.0002,
    densify_grad_threshold_fine_init = 0.0002,
    densify_grad_threshold_after = 0.0002,

    pruning_from_iter = 500,
    pruning_interval = 100,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005, 

    # TTGS: posemodel setting
    track_warmup_steps = 0,
    tool_fdm_warmup_steps = 0,

    track_position_lr_delay_mult = 0.01,
    track_rotation_lr_delay_mult = 0.01,
    track_position_max_steps = 3000, 
    track_rotation_max_steps = 3000, 

    track_position_lr_init =  0.05, 
    track_rotation_lr_init = 0.001,
    track_position_lr_final = 5.0e-5,
    track_rotation_lr_final = 1.0e-5,

    # TTGS: obj pose init
    obj_pose_init = 'cotrackerpnp', #'0' mean eye matrix init 
    percent_dense = 0.01,
    tool_percent_dense = 0.0,
    percent_big_ws = 0.1,
    tool_prune_big_points = True, 
    densify_grad_threshold_obj = 0.0002,
)

ModelHiddenParams = dict(
    curve_num = 17,
    ch_num = 10, # channel number of deformable attributes: 10 = 3 (scale) + 3 (mean) + 4 (rotation)
    init_param = 0.01, )

