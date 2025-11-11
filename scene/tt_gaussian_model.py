from scene.flexible_deform_model import TissueGaussianModel 
from scene.tool_model import ToolModel
from scene.tool_pose import ToolPose
# from scene.poses_all_objs import ActorPose
import torch.nn as nn
import torch
import os
from bidict import bidict
from utils.general_utils import matrix_to_quaternion,\
    startswith_any,strip_symmetric,build_scaling_rotation,quaternion_to_matrix
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p
from scene.cameras import Camera
from scene.gaussian_model_base import GaussianModelBase
from plyfile import PlyData, PlyElement
from typing import Union
from utils.general_utils import quaternion_raw_multiply

class TTGaussianModel(nn.Module):
    def __init__(self, metadata,new_cfg):
        super().__init__()
        self.cfg = new_cfg
        self.metadata = metadata
            
        self.max_sh_degree =self.cfg.model.gaussian.sh_degree
        self.active_sh_degree = self.max_sh_degree

        self.include_tissue =self.cfg.model.nsg.include_tissue #get('include_tissue', True)
        self.include_obj =self.cfg.model.nsg.include_obj#get('include_obj', False) #False)
        self.include_obj_pose =self.cfg.model.nsg.include_obj_pose#get('include_obj', False) #False)
        self.include_background =self.cfg.model.nsg.include_bkgd#get('include_bkgd', False)
        self.include_sky =self.cfg.model.nsg.include_sky#get('include_sky', False) 

        if self.include_sky:
            assert self.cfg.data.white_background is False
        # fourier sh dimensions
        self.fourier_dim =self.cfg.model.gaussian.get('fourier_dim', 1)
        # layer color correction
        self.use_color_correction =self.cfg.model.use_color_correction
        # camera pose optimizations (not test)
        self.use_pose_correction =self.cfg.model.use_pose_correction
        # symmetry
        self.flip_prob =self.cfg.model.gaussian.get('flip_prob', 0.)
        self.flip_axis = 1 
        self.flip_matrix = torch.eye(3).float().cuda() * -1
        self.flip_matrix[self.flip_axis, self.flip_axis] = 1
        self.flip_matrix = matrix_to_quaternion(self.flip_matrix.unsqueeze(0))
        # extend        
        self.disable_flip = True
        
        # extend
        # maintain this
        self.renderOnce = self.cfg.model.renderOnce
        # model names: manual add here!
        # save the same keys as self.model_name_id
        self.all_tool_objs_name = self.metadata['all_tool_objs_name']
        self.all_tool_objs_name_w_merged = self.metadata['all_tool_objs_name_w_merged']

        self.candidate_model_names = {}
        if self.include_background:
            self.candidate_model_names['bg_model'] = ['background']
            assert torch.Tensor([ name.startswith('background') for name in self.candidate_model_names['bg_model']]).all(),\
                f"not all names start_with background {self.candidate_model_names['bg_model']}"
            # assert len(self.candidate_model_names['bg_model'])==1,'later will use index[0]'
        if self.include_tissue:
            self.candidate_model_names['tissue_model'] = [
                                                        'tissue',
                                                        ]
            assert torch.Tensor([ name.startswith('tissue') for name in self.candidate_model_names['tissue_model']]).all(),\
                f"not all names start_with tissue {self.candidate_model_names['tissue_model']}"
        if self.include_obj:
            # track id of tool_model and pose_models follow the order here!--the order in the raw dataset
            self.candidate_model_names['obj_model_cand']= self.all_tool_objs_name #model_names_obj
        #/////////////////////////////
        self.setup_functions() 
        self.compo_all_gs_ordered_renderonce = self.cfg.model.compo_all_gs_ordered_renderonce
        # compo_all_gs_ordered_renderonce = controller.compo_all_gs_ordered_renderonce

        if self.renderOnce:   
            if self.compo_all_gs_ordered_renderonce == []:
                self.compo_all_gs_ordered_renderonce = self.candidate_model_names['obj_model_cand']+self.candidate_model_names['tissue_model']
            else:
                for model_name in self.compo_all_gs_ordered_renderonce:
                    assert model_name == 'tissue' or model_name.startswith('obj_tool'),f'{model_name} {self.compo_all_gs_ordered_renderonce}'
                    assert model_name in self.candidate_model_names['obj_model_cand']+self.candidate_model_names['tissue_model'],f'{model_name} {self.candidate_model_names["obj_model_cand"]}'
        else:
            pass
            # used for regularization with renderonce when train with render seperately
            self.compo_all_gs_ordered_renderonce = self.candidate_model_names['obj_model_cand']+self.candidate_model_names['tissue_model']


        self.transformed_tool_model_time_remark = -1 # used to control if the tool gs transformation have been done by check if the remark is the same as the viewcam.time

    def setup_functions(self):
        obj_tracklets = self.metadata['obj_tracklets']
        obj_info = self.metadata['obj_meta']
        tracklet_timestamps = self.metadata['tracklet_timestamps']
        camera_timestamps = self.metadata['camera_timestamps']
        
        self.model_name_id = bidict()
        self.obj_list = []
        self.obj_pose_list = []
        self.models_num = 0
        self.obj_info = obj_info
        # Build background model
        if self.include_background:
            model_names = self.candidate_model_names['bg_model']
            for model_name in model_names:
                model = GaussianModelBkgd(
                    model_name=model_name, 
                    scene_center=self.metadata['scene_center'],
                    scene_radius=self.metadata['scene_radius'],
                    sphere_center=self.metadata['sphere_center'],
                    sphere_radius=self.metadata['sphere_radius'],
                )
                setattr(self, model_name, model)
                self.model_name_id[model_name] = self.models_num
                self.models_num += 1

        # Build tissue model
        if self.include_tissue:
            model_names = self.candidate_model_names['tissue_model']
            for model_name in model_names:
                model = TissueGaussianModel(self.cfg.model.gaussian.sh_degree, \
                                                self.cfg.model.fdm)
                setattr(self, model_name, model )
                self.model_name_id[model_name] = self.models_num
                self.models_num += 1
        
        # Build object model
        self.poses_all_objs = None
        if self.include_obj:
            model_names = self.candidate_model_names['obj_model_cand']
            for i,model_name in enumerate(model_names):
                # ToolModel
                # fix a crutial bug here!! the 
                track_id = int(model_name.split('obj_tool')[-1])-1
                # track_id = i#int(model_name.split('obj_tool')[-1])-1
                assert track_id in [0,1,2,3],f'{model_name} {track_id}'
                from scene.tool_model import ToolModel
                model = ToolModel(model_args = self.cfg.model.gaussian,
                                  obj_meta=None,
                                #   track_id=i,
                                  track_id=track_id,
                                  cfg = self.cfg,
                                  disable_tool_fdm_scale=self.cfg.model.disable_tool_fdm_scale,
                                  )
                setattr(self, model_name, model)
                self.model_name_id[model_name] = self.models_num
                self.models_num += 1
                self.obj_list.append(model_name)
            # Build actor model 
            from scene.tool_pose import ToolPose
            if self.include_obj_pose:
                # camera_timestamps contains train and val
                # frames_num is complet continous imgs
                self.poses_all_objs = ToolPose(
                                                # objs_num=1, 
                                                objs_num=len(model_names),
                                                camera_timestamps=camera_timestamps, 
                                                cfg_optim=self.cfg.optim,
                                                # opt_track = self.cfg.model.nsg.opt_track,
                                                opt_track = self.cfg.model.use_opt_track,
                                                cam_id=0,
                                                cfg = self.cfg,
                                                tracklets=obj_tracklets,
                                                tracklet_timestamps=None)

                #disable tool pose learning
                disable_tool_pose_learning = True
                disable_tool_pose_learning = False
                if disable_tool_pose_learning:
                    # Freeze the first layer's weights
                    for param in self.poses_all_objs.parameters():
                        param.requires_grad = False

                self.obj_list.append(self.poses_all_objs)
                
    
    def set_visibility(self, include_list):
        self.include_list = include_list # prefix

    def get_visibility(self, model_name):
        if model_name.startswith('obj_'):
            if model_name in self.include_list and self.include_obj:
                return True
            else:
                # assert 0, f'{model_name} {self.include_list}'
                return False
        elif model_name.startswith('tissue'):
        # elif model_name == 'tissue':
            if model_name in self.include_list and self.include_tissue:
                return True
            else:
                return False
        else:
            raise ValueError(f'Unknown model name {model_name}')
    def create_from_pcd(self, 
                        pcd_dict: dict, 
                        spatial_lr_scale: float,\
                        time_line: int):# FDM need
        for model_name in self.model_name_id.keys():
            if model_name.startswith('tissue'):
                model: TissueGaussianModel = getattr(self, model_name)
                # Try to be the same at first
                model.create_from_pcd(pcd = pcd_dict[model_name], 
                                      spatial_lr_scale = spatial_lr_scale, 
                                      time_line = time_line)
            elif model_name.startswith('obj_'):
                model: ToolModel = getattr(self, model_name)
                model.create_from_pcd(pcd = pcd_dict[model_name], 
                                      spatial_lr_scale = spatial_lr_scale, 
                                      time_line = time_line)
            else:
                assert 0, model_name
    def save_ply(self, path, 
                 ):
        mkdir_p(os.path.dirname(path))
        plydata_list = []
        for i in range(self.models_num):
            model_name = self.model_name_id.inverse[i]
            model: GaussianModelBase = getattr(self, model_name)

            # if os.path.exists(path):
            #     assert 0,f'{model_name} {model} {path}'

            try:
                plydata = model.make_ply()
                plydata = PlyElement.describe(plydata, f'vertex_{model_name}')
            except:
                assert isinstance(model,TissueGaussianModel) or isinstance(model,ToolModel)
                plydata = model.save_ply(path = path,
                                         only_make = True)
                plydata = PlyElement.describe(plydata, f'vertex_{model_name}')
                
                # save ply model_wise for debug
                model.save_ply(path = path.replace('.ply',f'_{model_name}.ply'), only_make = False)



            plydata_list.append(plydata)
        PlyData(plydata_list).write(path)
        
    def load_ply(self, path):
        plydata_list = PlyData.read(path).elements
        for plydata in plydata_list:
            model_name = plydata.name[7:] # vertex_.....
            if model_name in self.model_name_id.keys():
                print('Loading model', model_name)
                model: Union[GaussianModelBase,TissueGaussianModel] = getattr(self, model_name)
                model.load_ply(path=None, 
                            #    input_ply=plydata,
                               input_ply = PlyData([plydata]),
                               )
                # plydata_list = PlyData.read(path).elements
        self.active_sh_degree = self.max_sh_degree
  
    def load_model(self, path):
        for model_name in self.model_name_id.keys():
            print('Loading model', model_name)
            model: Union[GaussianModelBase,TissueGaussianModel] = getattr(self, model_name)
            if isinstance(model,TissueGaussianModel):
                model.load_model(path)
            elif isinstance(model,ToolModel):
                pass
                if model.use_fdm_tool:
                    model.load_model(path)

            else:
                assert 0,model_name

        # try to load pose model
        if self.include_obj_pose:
            pose_model_path = os.path.join(path, 'pose_model.pth')
            # pose_model_path = pose_model_path.replace('300', '299')
            assert os.path.exists(pose_model_path)
            # load the pose model
            # print('' * 10, 'Loading pose model', pose_model_path,torch.load(pose_model_path).keys())
            if self.poses_all_objs.opt_track:
                self.poses_all_objs.load_state_dict(torch.load(pose_model_path)['params'])
        else:
            assert 0

    def load_state_dict(self, state_dict, exclude_list=[]):
        assert 0,'not tested'
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModelBase = getattr(self, model_name)
            model.load_state_dict(state_dict[model_name])
        
        if self.poses_all_objs is not None:
            self.poses_all_objs.load_state_dict(state_dict['poses_all_objs'])
                            
    def save_state_dict(self, is_final, exclude_list=[]):
        state_dict = dict()
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: Union[GaussianModelBase,TissueGaussianModel] = getattr(self, model_name)
            state_dict[model_name] = model.state_dict(is_final)
      
        return state_dict
    
    def set_camera_and_maintain_graph_obj_list(self,graph_obj_list, camera: Camera):
        ''''
        maintain: 
        self.graph_obj_list
        more ?
        '''
        # set camera
        self.viewpoint_camera = camera
        self.graph_obj_list = graph_obj_list#[] 
        for model_name in self.model_name_id.keys():
            if self.get_visibility(model_name=model_name):
                # model = getattr(self, model_name)
                if model_name.startswith('obj_'):
                    assert self.include_obj
                    assert model_name in self.graph_obj_list

    def transform_obj_pose(self, include_list):
        ego_pose = self.viewpoint_camera.ego_pose
        timestamp = self.viewpoint_camera.meta['timestamp']
        is_val_timestamp = self.viewpoint_camera.meta['is_val']
        cam = self.viewpoint_camera.meta['cam'][0]
        camera_timestamps = self.poses_all_objs.camera_timestamps[cam]['train_timestamps']
        
        assert "".join(self.graph_obj_list)=="".join(include_list)
        if len(self.graph_obj_list) > 0 and self.include_obj_pose:
            self.obj_rots = []
            self.obj_trans = []
            for i, obj_name in enumerate(self.graph_obj_list):
                obj_model: ToolModel = getattr(self, obj_name)
                track_id = obj_model.track_id
                assert track_id in [0,1,2],f'{obj_name} {track_id}'
                obj_rot = self.poses_all_objs.get_tracking_rotation(track_id, 
                                                                        # self.viewpoint_camera,
                                                                        timestamp=timestamp,
                                                                        is_val_timestamp=is_val_timestamp,
                                                                        camera_timestamps=camera_timestamps,
                                                                        ) 
                obj_trans = self.poses_all_objs.get_tracking_translation(track_id, 
                                                                        # self.viewpoint_camera,
                                                                        timestamp=timestamp,
                                                                        is_val_timestamp=is_val_timestamp,
                                                                        camera_timestamps=camera_timestamps,
                                                                        ) 
                # which learn the drift only--- 
                # ego_pose = self.viewpoint_camera.ego_pose
                ego_pose_rot = matrix_to_quaternion(ego_pose[:3, :3].unsqueeze(0)).squeeze(0)
                obj_rot = quaternion_raw_multiply(ego_pose_rot.unsqueeze(0), obj_rot.unsqueeze(0)).squeeze(0)
                obj_trans = ego_pose[:3, :3] @ obj_trans + ego_pose[:3, 3]
                
                obj_rot = obj_rot.expand(obj_model.get_xyz.shape[0], -1)
                obj_trans = obj_trans.unsqueeze(0).expand(obj_model.get_xyz.shape[0], -1)
                
                self.obj_rots.append(obj_rot)
                self.obj_trans.append(obj_trans)
            
            self.obj_rots = torch.cat(self.obj_rots, dim=0)
            self.obj_trans = torch.cat(self.obj_trans, dim=0)  
            
            self.flip_mask = []
            if not self.disable_flip:
                for obj_name in self.graph_obj_list:
                    obj_model: ToolModel = getattr(self, obj_name)
                    # assert 0, f"{obj_name}{obj_model.get_xyz}{obj_model.get_xyz.shape}"
                    flip_mask = torch.zeros_like(obj_model.get_xyz[:, 0]).bool()
                    self.flip_mask.append(flip_mask)
                self.flip_mask = torch.cat(self.flip_mask, dim=0)  

    @property
    def get_scaling(self):
        scalings = []
        for model_name in self.model_name_id.keys():
            if self.get_visibility(model_name=model_name):
                scaling = getattr(self, model_name).get_scaling
                scalings.append(scaling)
        scalings = torch.cat(scalings, dim=0)
        return scalings
            

    @property
    def get_rotation_obj_only(self):
        rotations = []
        # process obj pose
        rotations_local = []
        for i, obj_name in enumerate(self.graph_obj_list):
            assert self.get_visibility(model_name=obj_name), f'{obj_name} {self.include_list}'
            rotations_local.append(getattr(self, obj_name).get_rotation)
        if len(self.graph_obj_list) > 0:
            rotations_local = torch.cat(rotations_local, dim=0)
            rotations_local = rotations_local.clone()
            if not self.disable_flip:
                rotations_local[self.flip_mask] = quaternion_raw_multiply(self.flip_matrix, rotations_local[self.flip_mask])
            rotations_obj = quaternion_raw_multiply(self.obj_rots, rotations_local)
            rotations_obj = torch.nn.functional.normalize(rotations_obj)
            rotations.append(rotations_obj)

        rotations = torch.cat(rotations, dim=0)
        return rotations
    
    @property
    def get_xyz_obj_only(self):
        xyzs = []
        # # # process obj pose
        xyzs_local = []
        for i, obj_name in enumerate(self.graph_obj_list):
            assert self.get_visibility(model_name=obj_name), f'{obj_name} {self.include_list} {self.graph_obj_list}'
            xyzs_local.append(getattr(self, obj_name).get_xyz)
        if len(self.graph_obj_list) > 0:
            xyzs_local = torch.cat(xyzs_local, dim=0)
            xyzs_local = xyzs_local.clone()
            if not self.disable_flip:
                xyzs_local[self.flip_mask, self.flip_axis] *= -1
            obj_rots = quaternion_to_matrix(self.obj_rots)
            xyzs_obj = torch.einsum('bij, bj -> bi', obj_rots, xyzs_local) + self.obj_trans
            xyzs.append(xyzs_obj) 
    
        xyzs = torch.cat(xyzs, dim=0)
        return xyzs            

    


    @property
    def get_rotation(self):
        assert 0,'only implement below when use ttgs render'
        rotations = []
        for model_name in self.model_name_id.keys():
            if self.get_visibility(model_name=model_name):
                rotation = getattr(self, model_name).get_rotation
                rotations.append(rotation)

        # process obj pose
        rotations_local = []
        for i, obj_name in enumerate(self.graph_obj_list):
            assert self.get_visibility(model_name=obj_name)
            rotations_local.append(getattr(self, obj_name).get_rotation)
        if len(self.graph_obj_list) > 0:
            rotations_local = torch.cat(rotations_local, dim=0)
            rotations_local = rotations_local.clone()
            if not self.disable_flip:
                rotations_local[self.flip_mask] = quaternion_raw_multiply(self.flip_matrix, rotations_local[self.flip_mask])
            rotations_obj = quaternion_raw_multiply(self.obj_rots, rotations_local)
            rotations_obj = torch.nn.functional.normalize(rotations_obj)
            rotations.append(rotations_obj)

        rotations = torch.cat(rotations, dim=0)
        return rotations
    
    @property
    def get_xyz(self):
        assert 0,'only implement below when use ttgs render'
        # first tissue then obj(tool)
        xyzs = []
        for model_name in self.model_name_id.keys():
            if self.get_visibility(model_name=model_name):
                if isinstance(getattr(self, model_name),TissueGaussianModel):
                    xyz = getattr(self, model_name).get_xyz
                xyzs.append(xyz)

        # # # process obj pose
        xyzs_local = []
        for i, obj_name in enumerate(self.graph_obj_list):
            assert self.get_visibility(model_name=obj_name)
            xyzs_local.append(getattr(self, obj_name).get_xyz)
        if len(self.graph_obj_list) > 0:
            xyzs_local = torch.cat(xyzs_local, dim=0)
            xyzs_local = xyzs_local.clone()
            if not self.disable_flip:
                xyzs_local[self.flip_mask, self.flip_axis] *= -1
            obj_rots = quaternion_to_matrix(self.obj_rots)
            xyzs_obj = torch.einsum('bij, bj -> bi', obj_rots, xyzs_local) + self.obj_trans
            xyzs.append(xyzs_obj) 
    
        xyzs = torch.cat(xyzs, dim=0)
        return xyzs            
    @property
    def get_features(self):
        features = []
        for model_name in self.model_name_id.keys():
            if self.get_visibility(model_name=model_name):
                feature = getattr(self, model_name).get_features
                features.append(feature)
        features = torch.cat(features, dim=0)
        return features
    
    def get_colors(self, camera_center):
        assert 0, 'self.frame?'
        colors = []
        for model_name in self.model_name_id.keys():
            if self.get_visibility(model_name=model_name):
                model= getattr(self, model_name)
                max_sh_degree = model.max_sh_degree
                sh_dim = (max_sh_degree + 1) ** 2

                if model_name.startswith('tissue'):                  
                    shs = model.get_features.transpose(1, 2).view(-1, 3, sh_dim)
                elif model_name.startswith('obj_tool'):                  
                    features = model.get_features_fourier(self.frame)
                    shs = features.transpose(1, 2).view(-1, 3, sh_dim)
                else:
                    assert 0,model_name

                directions = model.get_xyz - camera_center
                directions = directions / torch.norm(directions, dim=1, keepdim=True)
                from utils.sh_utils import eval_sh
                sh2rgb = eval_sh(max_sh_degree, shs, directions)
                color = torch.clamp_min(sh2rgb + 0.5, 0.)
                colors.append(color)
        colors = torch.cat(colors, dim=0)
        return colors
                

    @property
    def get_opacity(self):
        opacities = []
        for model_name in self.model_name_id.keys():
            if self.get_visibility(model_name=model_name):
                opacity = getattr(self, model_name).get_opacity
                opacities.append(opacity)
        opacities = torch.cat(opacities, dim=0)
        return opacities
            
        
    def oneupSHdegree(self, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if model_name in exclude_list:
                continue
            model: Union[GaussianModelBase,TissueGaussianModel] = getattr(self, model_name)
            model.oneupSHdegree()
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, exclude_list=[]):
        self.active_sh_degree = 0
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: Union[GaussianModelBase,TissueGaussianModel,ToolModel] = getattr(self, model_name)
            if model_name.startswith('tissue'):
                model.training_setup(training_args=self.cfg.optim)
            elif model_name.startswith('obj_'):
                model.training_setup(training_args=self.cfg.optim)
            else:
                assert 0,NotImplementedError
                model.training_setup()
                
        if self.poses_all_objs is not None:
            self.poses_all_objs.training_setup()
        
    def update_learning_rate(self, iteration, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            # model: GaussianModelBase = getattr(self, model_name)
            model: Union[GaussianModelBase,TissueGaussianModel] = getattr(self, model_name)
            model.update_learning_rate(iteration)
        
        if self.poses_all_objs is not None:
            self.poses_all_objs.update_learning_rate(iteration)
    
    def update_optimizer(self, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: Union[GaussianModelBase,TissueGaussianModel] = getattr(self, model_name)
            model.update_optimizer()

        if self.poses_all_objs is not None:
            self.poses_all_objs.update_optimizer()
    
    def set_max_radii2D_all_models(self, radiis = {}, visibility_filters = {},model_names = []):
        '''
        already internnallly performed by the densify and prune of tissue model
        '''

        assert len(radiis.keys()) == len(visibility_filters.keys())
        assert len(radiis.keys()) == len(model_names),f'{len(radiis.keys())}{len(model_names)} {radiis.keys()}{model_names}'

        try:
            for model_name in model_names:
                visibility_filter = visibility_filters[model_name]
                radii = radiis[model_name]
                radii = radii.float()
                model: GaussianModelBase = getattr(self, model_name)
                model.max_radii2D[visibility_filter] = torch.max(
                    model.max_radii2D[visibility_filter], radii[visibility_filter])
        except:
            assert 0, f'{visibility_filters.keys()} {model_names} {radiis.keys()} \
                {model_name} {model.max_radii2D.shape} {radii.shape} {visibility_filter.shape}'

    
    def add_densification_stats_all_models(self, viewspace_point_tensors= {}, visibility_filters= {}, model_names = [],
                                           compo_all_gs_ordered_idx = None,):
        '''
        already internnallly performed by the densify and prune of tissue model
        '''

        assert len(visibility_filters.keys()) == len(model_names)
        if isinstance(viewspace_point_tensors,dict):
            render_once = False
        else:
            render_once = True
            assert compo_all_gs_ordered_idx is not None

        if render_once:
            all_viewspace_point_tensors_grad = viewspace_point_tensors.grad
        for model_name in model_names:
            visibility_filter = visibility_filters[model_name]
            if render_once:
                start_idx, end_idx = compo_all_gs_ordered_idx[model_name]
                viewspace_point_tensor_grad = all_viewspace_point_tensors_grad[start_idx:(end_idx+1)]
            else:
                assert len(viewspace_point_tensors.keys()) == len(visibility_filters.keys()),f"{viewspace_point_tensors.keys()} {visibility_filters.keys()}"
                viewspace_point_tensor = viewspace_point_tensors[model_name]
                viewspace_point_tensor_grad = viewspace_point_tensor.grad
            model: GaussianModelBase = getattr(self, model_name)
            assert viewspace_point_tensor_grad!=None
            model.xyz_gradient_accum[visibility_filter, 0:1] += torch.norm(viewspace_point_tensor_grad[visibility_filter, :2], dim=-1, keepdim=True)
            model.xyz_gradient_accum[visibility_filter, 1:2] += torch.norm(viewspace_point_tensor_grad[visibility_filter, 2:], dim=-1, keepdim=True)
            model.denom[visibility_filter] += 1



    
    def set_max_radii2D_all_models_v0(self, radiis = [], visibility_filters = [],model_names = []):
        '''
        already internnallly performed by the densify and prune of tissue model
        '''

        assert len(radiis) == len(visibility_filters)
        assert len(radiis) == len(model_names)

        for radii, visibility_filter, model_name in zip(radiis,visibility_filters,model_names):
            radii = radii.float()
            model: GaussianModelBase = getattr(self, model_name)
            model.max_radii2D[visibility_filter] = torch.max(
                model.max_radii2D[visibility_filter], radii[visibility_filter])

    def add_densification_stats_all_models_v0(self, viewspace_point_tensors, visibility_filters, model_names):
        '''
        already internnallly performed by the densify and prune of tissue model
        '''

        assert len(viewspace_point_tensors) == len(model_names)
        assert len(visibility_filters) == len(model_names)


        for viewspace_point_tensor, visibility_filter, model_name in zip(viewspace_point_tensors,visibility_filters,model_names):
            # assert 0,'not checked'
            viewspace_point_tensor_grad = viewspace_point_tensor.grad
            model: GaussianModelBase = getattr(self, model_name)
            model.xyz_gradient_accum[visibility_filter, 0:1] += torch.norm(viewspace_point_tensor_grad[visibility_filter, :2], dim=-1, keepdim=True)
            model.xyz_gradient_accum[visibility_filter, 1:2] += torch.norm(viewspace_point_tensor_grad[visibility_filter, 2:], dim=-1, keepdim=True)
            model.denom[visibility_filter] += 1

    
    
    def reset_opacity(self, exclude_list=[]):
        # assert 0
        '''
        already internnallly performed by the densify and prune of tissue model
        '''
        for model_name in self.model_name_id.keys():
            if model_name not in self.compo_all_gs_ordered_renderonce:
                continue

            model: GaussianModelBase = getattr(self, model_name)
            if startswith_any(model_name, exclude_list):
                continue
            model.reset_opacity()


    def densify_and_prune(self, 
                          max_grad = None, 
                          min_opacity = None,
                          exclude_list=[],
                          extent = None,
                          max_screen_size = None,
                          skip_densify = None,
                          skip_prune = None,
                          percent_big_ws = None,
                          current_iter = None,
                          max_iter = None,
                          densify_until_iter_tool = None,
                          ):
        scalars = {}#None
        tensors = {}#None

        for model_name in self.compo_all_gs_ordered_renderonce:
            if startswith_any(model_name, exclude_list):
                continue
            # if model_name == 'tissue':
            model: Union[TissueGaussianModel,ToolModel] = getattr(self, model_name)
            if isinstance(model,TissueGaussianModel) \
                    :

                scalars_, tensors_ = model.densify_and_prune(max_grad = max_grad, 
                                                            min_opacity = min_opacity, 
                                                            extent=extent, 
                                                            max_screen_size=max_screen_size,
                                                            skip_densify=skip_densify,
                                                            skip_prune=skip_prune,
                                                            )
            elif isinstance(model,ToolModel):
                if current_iter > densify_until_iter_tool:
                    continue

                assert model_name.startswith('obj_tool'),model_name
                current_tool_mask = torch.Tensor(getattr(self.viewpoint_camera,f'raw_{model_name}')).squeeze(0)
                within_which_2d=torch.Tensor(self.metadata['init_mask_dict'][model_name]).to(torch.bool)
                means3D_final = None
                if '2Dcurrent' in self.cfg.optim.tool_adc_mode or self.cfg.model.dbg_vis_adc:

                    bg_color = [1, 1, 1] if self.cfg.data.white_background else [0, 0, 0]
                    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                    from gaussian_renderer import ttgs_render
                    _,_,means3D_final = ttgs_render(self.viewpoint_camera, 
                                                    getattr(self,model_name),
                                                    self.cfg.render, 
                                                background,
                                                debug_getxyz_ttgs=True,
                                                ttgs_model=self,
                                                single_compo_or_list=model_name,
                                                tool_parse_cam_again = True,
                                                vis_img_debug = self.cfg.render.dbg_vis_render,
                                                vis_img_debug_title_more = 'PRUNE_RENDER: current strategy',

                                                )

                scalars_, tensors_ = model.densify_and_prune(max_grad = max_grad, 
                                                            min_opacity = min_opacity, 
                                                            extent=extent, 
                                                            K=torch.Tensor(self.viewpoint_camera.K),
                                                            within_which_2d=within_which_2d,
                                                            current_tool_mask = current_tool_mask,
                                                            dbg_vis_tool_adc = self.cfg.model.dbg_vis_adc ,
                                                            skip_densify=skip_densify,
                                                            skip_prune=skip_prune,
                                                            tool_adc_mode = self.cfg.optim.tool_adc_mode,                                                            
                                                            means3D_final = means3D_final,
                                                            current_iter=current_iter,
                                                            max_iter=max_iter,
                                                            do_current_adc_since_iter = self.cfg.optim.do_current_adc_since_iter,
                                                            )
            if model_name == 'background':
                scalars = scalars_
                tensors = tensors_
    
        return scalars, tensors
    

    def get_box_reg_loss(self):
        box_reg_loss = 0.
        for obj_name in self.obj_list:
            assert 0
            obj_model: ToolPose = getattr(self, obj_name)
            box_reg_loss += obj_model.box_reg_loss()
        box_reg_loss /= len(self.obj_list)

        return box_reg_loss
            
