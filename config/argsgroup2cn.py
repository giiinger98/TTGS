
import os

def perform_args2cfg(args,remain_redundant,dbg_print):
    from argparse import ArgumentParser

    # remain_redundant = False
    #///////////////////////////////////////////////////////////////
    #convert our args param to cfg format for less changing their code
    from config.argsgroup_ttgs import EvalParams,TrainParams,OptParams,ModParams,DataParams,RenderParams,ViewerParams
    from config.argsgroup_ttgs import OTHER_PARAM_DICT
    from config.argsgroup2cn import group_params_to_cfgnode
    # reset parser as the previous were used for obatin args
    parser = ArgumentParser(description="Training script parameters")
    eval_stree = EvalParams(parser)
    train_stree = TrainParams(parser)
    opt_stree = OptParams(parser)
    mod_stree = ModParams(parser)
    data_stree = DataParams(parser)
    render_stree = RenderParams(parser)
    viewer_stree = ViewerParams(parser)
    # dataset, hyper, opt, pipe = lp.extract(args), hp.extract(args), op.extract(args),pp.extract(args)
    eval_stree_param,train_stree_param,opt_stree_param,\
        mod_stree_param,data_stree_param,render_stree_param,viewer_stree_param = [None]*7
    # update the _param based on args and (default)
    # group instance: only resetted ((the default_inited + resetted
    eval_stree_param,*missing_resetted_redundant = eval_stree.extract(args ,remain_redundant = remain_redundant, dbg=dbg_print)
    train_stree_param,*missing_resetted_redundant = train_stree.extract(args ,remain_redundant = remain_redundant, dbg=dbg_print)
    opt_stree_param,*missing_resetted_redundant = opt_stree.extract(args ,remain_redundant = remain_redundant, dbg=dbg_print)
    #special: internnal contain confignode
    mod_stree_param,*missing_resetted_redundant = mod_stree.extract(args ,remain_redundant = remain_redundant, dbg=dbg_print)
    data_stree_param,*missing_resetted_redundant = data_stree.extract(args ,remain_redundant = remain_redundant, dbg=dbg_print)
    render_stree_param,*missing_resetted_redundant = render_stree.extract(args ,remain_redundant = remain_redundant, dbg=dbg_print)
    viewer_stree_param,*missing_resetted_redundant = viewer_stree.extract(args ,remain_redundant = remain_redundant, dbg=dbg_print)
    other_param_dict = OTHER_PARAM_DICT
    #prepare cfg
    #confignode instance
    cfg = group_params_to_cfgnode(inputParam=eval_stree_param,groupParam_name='eval',cfg=None)#create parent CN
    cfg = group_params_to_cfgnode(inputParam=train_stree_param,groupParam_name='train',cfg=cfg)
    cfg = group_params_to_cfgnode(inputParam=opt_stree_param,groupParam_name='optim',cfg=cfg)
    cfg = group_params_to_cfgnode(inputParam=mod_stree_param,groupParam_name='model',cfg=cfg)
    cfg = group_params_to_cfgnode(inputParam=data_stree_param,groupParam_name='data',cfg=cfg)
    cfg = group_params_to_cfgnode(inputParam=render_stree_param,groupParam_name='render',cfg=cfg)
    cfg = group_params_to_cfgnode(inputParam=viewer_stree_param,groupParam_name='viewer',cfg=cfg)
    cfg = group_params_to_cfgnode(inputParam=other_param_dict,groupParam_name = None, cfg=cfg)
    #save the cfg file--hard code
    cfg.expname = args.expname # extend    
    cfg.model_path = args.expname # extend    
    cfg.data.source_path = args.source_path
    cfg.data.model_path = args.model_path
    cfg.data.type = 'endonerf'
    #copied from the parse_cfg internally 
    cfg.trained_model_dir = os.path.join(cfg.model_path, 'trained_model')
    cfg.point_cloud_dir = os.path.join(cfg.model_path, 'point_cloud')
    return cfg, (eval_stree_param,train_stree_param,opt_stree_param,\
        mod_stree_param,data_stree_param,render_stree_param,viewer_stree_param,\
            other_param_dict)



def save_cfg(cfg, model_dir, epoch=0):
    from contextlib import redirect_stdout
    os.system('mkdir -p {}'.format(model_dir))
    cfg_dir = os.path.join(model_dir, 'configs')
    os.system('mkdir -p {}'.format(cfg_dir))

    cfg_path = os.path.join(cfg_dir, f'config_{epoch:06d}.yaml')
    with open(cfg_path, 'w') as f:
        with redirect_stdout(f): print(cfg.dump())
        
    print(f'Save input config to {cfg_path}')


def group_params_to_cfgnode(inputParam, groupParam_name = None,cfg = None):

    """
    # #usage
    # cfg = group_params_to_cfgnode(inputParam=opt,groupParam_name='optim',cfg=None)
    # cfg = group_params_to_cfgnode(inputParam=pipe,groupParam_name=['middle','child1'],cfg=cfg)
    # cfg = group_params_to_cfgnode(inputParam=pipe,groupParam_name=['middle','child2'],cfg=cfg)#will overwrite
    # cfg = group_params_to_cfgnode(inputParam={'hah':23,'b':333,},groupParam_name = None, cfg=cfg)

    
    """
    from config.yacs import CfgNode as CN
    # from arguments import GroupParams
    from config.argsgroup_ttgs import GroupParams
    # Create an empty parent CN if not exist
    cfg = CN() if cfg == None else cfg
    # Create an empty child CN if name for the child is parsed
    if groupParam_name != None:
        cfg_child = CN() 
        assert isinstance(inputParam,GroupParams),'we only do this when parsing GroupParams for cleaness'
    else:
        cfg_child = None
    # Iterate over opt's attributes and add them to cfg
    if isinstance(inputParam,GroupParams):
        for key, value in inputParam.__dict__.items():
            if isinstance(value, dict):  # If the value is a dictionary, convert it to a nested CN
                if groupParam_name == None:
                    cfg[key] = CN(value) 
                else:
                    # assert 0, f'the current groupParam not have this...'
                    cfg_child[key] = CN(value) 
            else:
                if groupParam_name == None:
                    cfg[key] = value  # Add the value directly
                else:
                    cfg_child[key] = value 

        if groupParam_name != None:
            assert cfg_child!=None
            if isinstance(groupParam_name,list):
                # resursive--espically for the stree ModelParam
                assert len(groupParam_name)==2, NotImplementedError
                middle_name,child_name = groupParam_name
                try:
                    cfg_middle = cfg[middle_name]
                    cfg_middle[child_name] = cfg_child
                except:
                    cfg_middle = CN()
                    cfg_middle[child_name] = cfg_child
                    cfg[middle_name] = cfg_middle
            else:

                cfg[groupParam_name] = cfg_child
    elif isinstance(inputParam,dict):
        for key, value in inputParam.items():
            cfg[key] = value 
    elif inputParam== None:
        pass
    else:
        assert 0
    return cfg