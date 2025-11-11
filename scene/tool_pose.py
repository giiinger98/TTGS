import torch
import torch.nn as nn
import numpy as np
from utils.general_utils import quaternion_raw_multiply, get_expon_lr_func, quaternion_slerp, matrix_to_quaternion,quaternion_to_matrix
from utils.camera_utils import Camera

class ToolPose(nn.Module):      
    def __init__(self, 
                #  frames_num,
                 objs_num = 1,
                 camera_timestamps = None, 
                #  obj_info, 
                 cfg_optim = None,
                 opt_track = True,
                 cam_id = 0,
                 cfg = None,
                 tracklets = None,
                 tracklet_timestamps = None,
                 ):
        # tracklets: [num_frames, max_obj, [track_id, x, y, z, qw, qx, qy, qz]]
        # frame_timestamps: [num_frames]
        super().__init__()
        self.cfg = cfg
        self.cfg_optim = cfg_optim
        self.camera_timestamps = camera_timestamps
        self.timestamps = self.camera_timestamps[str(cam_id)]['all_timestamps']
        # we predict abs pose
        frames_num = len(self.timestamps)
        # obj_pose_rot_optim_space = 'rpy', #'lie'
        # assert self.objs_num == 1,self.objs_num
        # self.objs_num = 2
        self.objs_num = objs_num
        if self.cfg_optim.obj_pose_init == '0':
            self.input_trans = torch.zeros([frames_num,self.objs_num,3]).float().cuda()
            self.input_rots_quat = torch.zeros([frames_num,self.objs_num,4]).float().cuda() #wxyz
            self.input_rots_quat[:,:,0] = 1
        elif self.cfg_optim.obj_pose_init == 'cotrackerpnp':
            self.input_trans = torch.zeros([frames_num,self.objs_num,3]).float().cuda()
            self.input_rots_quat = torch.zeros([frames_num,self.objs_num,4]).float().cuda() #wxyz
            self.input_rots_quat[:,:,0] = 1
            for i in range(self.objs_num):
                self.cotrackerpnp_trajectory_cams2w = tracklets[f'obj_tool{i+1}']['trajectory_cams2w'].float().cuda()# 
                load_num,_,_ = self.cotrackerpnp_trajectory_cams2w.shape
                assert load_num == frames_num
                cotrackerpnp_trajectory_w2cams2 = torch.linalg.inv(self.cotrackerpnp_trajectory_cams2w)
                self.input_trans[:,i,:] = cotrackerpnp_trajectory_w2cams2[:,:3,3]
                input_rots_mat = cotrackerpnp_trajectory_w2cams2[:,:3,:3]
                self.input_rots_quat[:,i,:] = matrix_to_quaternion(input_rots_mat)#self.cotrackerpnp_trajectory_cams2w[:,:3,3]      
        else:
            assert 0,  self.cfg_optim.obj_pose_init

        self.opt_track = opt_track #cfg.model.nsg.opt_track
        # if self.opt_track:
        # if self.cfg_optim.obj_pose_rot_optim_space == 'rpy':
        self.opt_trans = nn.Parameter(torch.zeros_like(self.input_trans)).requires_grad_(True).to(self.input_trans.device) 
        f_num,obj_num,_  = self.opt_trans.shape
        self.opt_rots_rpy = nn.Parameter(torch.zeros([f_num,obj_num,3],
                                                        device = self.input_trans.device)).requires_grad_(True)\
                                                        .to(self.opt_trans.device).to(self.opt_trans.dtype)  

        # maintained...        
        self.cotrackerpnp_trajectory_cams2w_corrected = torch.zeros([frames_num,4,4]).float().cuda()
        self.correct_abs_input_trans = torch.zeros([frames_num,self.objs_num,3]).float().cuda()
        self.correct_abs_input_rots_quat = torch.zeros([frames_num,self.objs_num,4]).float().cuda()
        self.correct_abs_input_rots_quat[:,:,0] = 1
        
        # self.
        # else:
            # self.opt_transform_lie = None
            # assert 0,  self.cfg_optim.obj_pose_rot_optim_space
        

            # assert 0, NotImplementedError

    def training_setup(self):
        if self.opt_track:
            params = [
                {'params': [self.opt_trans], 'lr': self.cfg_optim.track_position_lr_init, 'name': 'opt_trans'},
                {'params': [self.opt_rots_rpy], 'lr': self.cfg_optim.track_rotation_lr_init, 'name': 'opt_rots_rpy'},
                # {'params': [self.opt_rots_mat], 'lr': self.cfg_optim.track_rotation_lr_init, 'name': 'opt_rots_mat'},
            ]
            # assert 0,self.cfg_optim.track_position_lr_init
            self.opt_trans_scheduler_args = get_expon_lr_func(lr_init=self.cfg_optim.track_position_lr_init,
                                                    lr_final=self.cfg_optim.track_position_lr_final,
                                                    lr_delay_mult=self.cfg_optim.track_position_lr_delay_mult,
                                                    # max_steps=self.cfg.train.iterations,
                                                    max_steps=self.cfg_optim.track_position_max_steps,
                                                    # warmup_steps=self.cfg_optim.opacity_reset_interval,
                                                    warmup_steps=self.cfg_optim.track_warmup_steps,
                                                    disable = False,
                                                    )
            
            self.opt_rots_rpy_scheduler_args = get_expon_lr_func(lr_init=self.cfg_optim.track_rotation_lr_init,
                                                    lr_final=self.cfg_optim.track_rotation_lr_final,
                                                    lr_delay_mult=self.cfg_optim.track_rotation_lr_delay_mult,
                                                    # max_steps=self.cfg.train.iterations,
                                                    max_steps=self.cfg_optim.track_rotation_max_steps,
                                                    # warmup_steps=self.cfg_optim.opacity_reset_interval,
                                                    warmup_steps=self.cfg_optim.track_warmup_steps,
                                                    disable = False,
                                                    
                                                    ) 

            # self.opt_rots_mat_scheduler_args = get_expon_lr_func(lr_init=self.cfg_optim.track_rotation_lr_init,
            #                                         lr_final=self.cfg_optim.track_rotation_lr_final,
            #                                         lr_delay_mult=self.cfg_optim.track_rotation_lr_delay_mult,
            #                                         max_steps=self.cfg_optim.track_rotation_max_steps,
            #                                         warmup_steps=self.cfg_optim.opacity_reset_interval)    
            
            self.optimizer = torch.optim.Adam(params=params, lr=0, eps=1e-15)
        else:
            pass
            # assert 0
    
    def update_learning_rate(self, iteration):
        if self.opt_track:
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "opt_trans":
                    lr = self.opt_trans_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "opt_rots_rpy":
                    lr = self.opt_rots_rpy_scheduler_args(iteration)
                    param_group['lr'] = lr
                # if param_group["name"] == "opt_rots_mat":
                #     lr = self.opt_rots_mat_scheduler_args(iteration)
                #     param_group['lr'] = lr
        
    def update_optimizer(self):
        if self.opt_track:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=None)
        
    def find_closest_camera_timestamps(self, track_id, 
                                    timestamp,
                                    camera_timestamps,   
                                    # camera: Camera,
                                       ):
        # timestamp = camera.meta['timestamp']
        # cam = camera.meta['cam'][0]
        # camera_timestamps = self.camera_timestamps[cam]['train_timestamps']
        try:
            start_timestamp = self.obj_info[track_id]['start_timestamp']
            end_timestamp = self.obj_info[track_id]['end_timestamp']
            camera_timestamps = np.array([x for x in camera_timestamps if x >= start_timestamp and x <= end_timestamp])
        except:
            camera_timestamps = np.array([x for x in camera_timestamps])


        if len(camera_timestamps) < 2:
            return None, None
        else:
            delta_timestamps = np.abs(camera_timestamps - timestamp)
            idx1, idx2 = np.argsort(delta_timestamps)[:2]
            # print(f'*********test_tgt{timestamp} nbt_in_trn{idx1}:{camera_timestamps[idx1]} {idx2}:{camera_timestamps[idx2]} {delta_timestamps}')            
            return camera_timestamps[idx1], camera_timestamps[idx2]

    def find_closest_camera_timestamps_ori(self, track_id, camera: Camera):
        timestamp = camera.meta['timestamp']
        cam = camera.meta['cam'][0]
        camera_timestamps = self.camera_timestamps[cam]['train_timestamps']
        try:
            start_timestamp = self.obj_info[track_id]['start_timestamp']
            end_timestamp = self.obj_info[track_id]['end_timestamp']
            camera_timestamps = np.array([x for x in camera_timestamps if x >= start_timestamp and x <= end_timestamp])
        except:
            camera_timestamps = np.array([x for x in camera_timestamps])


        if len(camera_timestamps) < 2:
            return None, None
        else:
            delta_timestamps = np.abs(camera_timestamps - timestamp)
            idx1, idx2 = np.argsort(delta_timestamps)[:2]
            # print(f'*********test_tgt{timestamp} nbt_in_trn{idx1}:{camera_timestamps[idx1]} {idx2}:{camera_timestamps[idx2]} {delta_timestamps}')            
            return camera_timestamps[idx1], camera_timestamps[idx2]


    def get_tracking_rotation(self, track_id,
                                 timestamp = None,
                                 is_val_timestamp = False,
                                 camera_timestamps = None,
                                 camera = None,):
        if camera is not None:
            timestamp = camera.meta['timestamp']
            is_val_timestamp = camera.meta['is_val']
            cam = camera.meta['cam'][0]
            camera_timestamps = self.camera_timestamps[cam]['train_timestamps']
        else:
            assert timestamp is not None
            assert camera_timestamps is not None



        # if self.opt_track and camera.meta['is_val']:
        if self.opt_track and is_val_timestamp:

            # return self.get_tracking_rotation_(track_id, camera.meta['timestamp'])
            timestamp1, timestamp2 = self.find_closest_camera_timestamps(track_id,
                                                                            timestamp = timestamp,
                                                                            camera_timestamps = camera_timestamps, 
                                                                         )
            if timestamp1 is None:
                return self.get_tracking_rotation_(track_id, timestamp)
            else:
                # timestamp = camera.meta['timestamp']
                rots1 = self.get_tracking_rotation_(track_id, timestamp1)
                rots2 = self.get_tracking_rotation_(track_id, timestamp2)
                r = (timestamp - timestamp1) / (timestamp2 - timestamp1)
                rots = quaternion_slerp(rots1, rots2, r)
                return rots
        else:
            # print('//////////////TRN POSE TRANS INTERPOLATION')
            return self.get_tracking_rotation_(track_id, timestamp)



    def get_tracking_translation(self, track_id,
                                 timestamp = None,
                                 is_val_timestamp = False,
                                 camera_timestamps = None,
                                 camera = None,):
        '''
        SUPPORT PARSE CAMERA DIRECLY OR NEEDED CAMERA ATTRIBUTES
        '''
        if camera is not None:
            assert timestamp== None
            assert camera_timestamps==None
            timestamp = camera.meta['timestamp']
            is_val_timestamp = camera.meta['is_val']
            cam = camera.meta['cam'][0]
            camera_timestamps = self.camera_timestamps[cam]['train_timestamps']
        else:
            assert timestamp is not None
            assert camera_timestamps is not None

        # if self.opt_track and camera.meta['is_val']:
        if self.opt_track and is_val_timestamp:
            # return self.get_tracking_translation_(track_id, camera.meta['timestamp'])

            timestamp1, timestamp2 = self.find_closest_camera_timestamps(track_id, 
                                                                        timestamp = timestamp,
                                                                        camera_timestamps = camera_timestamps, 
                                                                         )
            if timestamp1 is None:
                return self.get_tracking_translation_(track_id, timestamp)
            else:
                # timestamp = camera.meta['timestamp']
                trans1 = self.get_tracking_translation_(track_id, timestamp1)
                trans2 = self.get_tracking_translation_(track_id, timestamp2)
                trans = (trans1 * (timestamp2 - timestamp) + trans2 * (timestamp - timestamp1)) / (timestamp2 - timestamp1)
                return trans
            
        else:
            return self.get_tracking_translation_(track_id,timestamp)


    def get_tracking_translation_(self, track_id, 
                                  cam_timestamp,
                                #   camera: Camera,
                                  ):
        # assert track_id==0
        assert track_id in [0,1,2], track_id
        # cam_timestamp = camera.meta['timestamp']
        frame_idx = self.timestamps.index(cam_timestamp)
        # return self.input_trans[frame_idx, track_id]
        trans = self.opt_trans[frame_idx, track_id] 
        # print(f'debug opt_trans {frame_idx}: all_0?{not trans.any()} {trans}')
        # if frame_idx == 24:
            # pass
        # print(f"debug ***********opt_trans{len(self.opt_trans)}",frame_idx,"delta",self.opt_trans[-3:],"input",self.input_trans[-3:])
        trans = trans + self.input_trans[frame_idx, track_id] 
 
        return trans



    def get_tracking_rotation_(self, track_id, 
                                cam_timestamp
                               ):
        '''
        param to learn is rpy
        return in wxyz format(gs)'''
        
        import math
        def euler_to_quaternion(roll_pitch_yaw):
            """
            Convert Euler angles (roll, pitch, yaw) to a quaternion (w, x, y, z).
            Roll, pitch, and yaw should be in radians.
            """
            assert roll_pitch_yaw.shape == torch.Size([3])
            roll,pitch,yaw = roll_pitch_yaw
            # Compute half angles
            cy = math.cos(yaw * 0.5)
            sy = math.sin(yaw * 0.5)
            cp = math.cos(pitch * 0.5)
            sp = math.sin(pitch * 0.5)
            cr = math.cos(roll * 0.5)
            sr = math.sin(roll * 0.5)

            # Compute quaternion
            w = cr * cp * cy + sr * sp * sy
            x = sr * cp * cy - cr * sp * sy
            y = cr * sp * cy + sr * cp * sy
            z = cr * cp * sy - sr * sp * cy

            return torch.Tensor([w,x,y,z]).to(roll_pitch_yaw.device)
            return w, x, y, z

        def euler_to_quaternion_torch(roll_pitch_yaw):
            """
            Convert Euler angles (roll, pitch, yaw) to a quaternion (w, x, y, z).
            Roll, pitch, and yaw should be in radians.
            """
            assert roll_pitch_yaw.shape == torch.Size([3])
            roll,pitch,yaw = roll_pitch_yaw
            # Compute half angles
            cy = torch.cos(yaw * 0.5)
            sy = torch.sin(yaw * 0.5)
            cp = torch.cos(pitch * 0.5)
            sp = torch.sin(pitch * 0.5)
            cr = torch.cos(roll * 0.5)
            sr = torch.sin(roll * 0.5)

            # Compute quaternion
            w = cr * cp * cy + sr * sp * sy
            x = sr * cp * cy - cr * sp * sy
            y = cr * sp * cy + sr * cp * sy
            z = cr * cp * sy - sr * sp * cy

            # quant = torch.zeros_like(torch.Tensor([w,x,y,z])).to(roll_pitch_yaw.device)
            # quant.requires_grad = True
            # quant[0:4] = torch.Tensor([w,x,y,z]).to(roll_pitch_yaw.device)
            # return quant
            return torch.stack((w, x, y, z), dim=-1)
            return torch.Tensor([w,x,y,z]).to(roll_pitch_yaw.device)
            return w, x, y, z


        # Example usage:
        # roll = 0.1  # radians
        # pitch = 0.2  # radians
        # yaw = 0.3  # radians
        # assert track_id==0
        assert track_id in [0,1,2],f'{track_id}'
        # assert 0,self.opt_rots_rpy.shape
        # cam_timestamp = camera.meta['timestamp']
        frame_idx = self.timestamps.index(cam_timestamp)
        roll_pitch_yaw = self.opt_rots_rpy[frame_idx,track_id]
        quaternion = euler_to_quaternion_torch(roll_pitch_yaw)
        quaternion_input = self.input_rots_quat[frame_idx,track_id]
        # if frame_idx == 24:
            # print("debug ***********opt_rots_rpy",frame_idx,"delta",quaternion.data,"input",quaternion_input.data)
        quaternion = quaternion_raw_multiply(quaternion_input.unsqueeze(0), 
                                    quaternion.unsqueeze(0)).squeeze(0)
        # print("Quaternion (w, x, y, z):", quaternion)
        return quaternion

    
    def reset_delta_pose(self):
        # self.input_trans = self.correct_abs_input_trans#[frame_idx,track_id] = self.get_tracking_translation(track_id, camera)
        # self.input_rots_quat = self.correct_abs_input_rots_quat    
        self.opt_trans = nn.Parameter(torch.zeros_like(self.input_trans)).requires_grad_(True).to(self.input_trans.device) 
        f_num,obj_num,_  = self.opt_trans.shape
        self.opt_rots_rpy = nn.Parameter(torch.zeros([f_num,obj_num,3],
                                                        device = self.input_trans.device)).requires_grad_(True)\
                                                        .to(self.opt_trans.device).to(self.opt_trans.dtype)  



    def update_input_pose_with_learned(self):
        self.input_trans = self.correct_abs_input_trans#[frame_idx,track_id] = self.get_tracking_translation(track_id, camera)
        self.input_rots_quat = self.correct_abs_input_rots_quat

    def save_learned_tool_pose(self, video_all_Cameras,saved_dir = None,disable_grad = True):
        # camera.meta['timestamp']
        for camera in video_all_Cameras:
            frame_idx = camera.meta['timestamp']
            for track_id in range(self.objs_num):
                # roll_pitch_yaw = self.opt_rots_rpy[frame_idx,track_id]
                self.correct_abs_input_trans[frame_idx,track_id] = self.get_tracking_translation(track_id, 
                                                    camera = camera).detach()  if not disable_grad else self.get_tracking_translation(track_id, camera = camera)
                self.correct_abs_input_rots_quat[frame_idx,track_id] = self.get_tracking_rotation(track_id, 
                                                    camera = camera).detach() if not disable_grad else self.get_tracking_rotation(track_id, camera = camera)  

        if saved_dir is not None:
            self.save_learned_tool_pose_as_tracklet_(saved_dir)

    def save_learned_tool_pose_as_tracklet_(self,saved_dir):
        # camera.meta['timestamp']
        # frame_idx = camera.meta['timestamp']
        # for track_id in range(self.objs_num):
        # roll_pitch_yaw = self.opt_rots_rpy[frame_idx,track_id]
        # self.correct_abs
        assert self.correct_abs_input_trans.shape[0] == self.correct_abs_input_rots_quat.shape[0]
        assert self.correct_abs_input_trans.shape[1] == self.correct_abs_input_rots_quat.shape[1]
        learned_tracklets = {}
        # learned_trajectory_cams2w
        # learned_trajectory_w2cams2 = torch.zeros_like(self.cotrackerpnp_trajectory_cams2w).to(self.cotrackerpnp_trajectory_cams2w.dtype)
        # learned_trajectory_w2cams2 = torch.zeros_like(self.cotrackerpnp_trajectory_cams2w).to(self.cotrackerpnp_trajectory_cams2w.dtype)
        learned_trajectory_w2cams2 = torch.zeros_like(self.cotrackerpnp_trajectory_cams2w_corrected).to(self.cotrackerpnp_trajectory_cams2w_corrected.dtype)
        # self.cotrackerpnp_trajectory_cams2w_corrected = torch.zeros([frames_num,4,4]).float().cuda()
        # 
        learned_trajectory_w2cams2[:,3,3] = 1
        for track_id in range(self.objs_num):
            trans,rot_quat  = self.correct_abs_input_trans[:,track_id],self.correct_abs_input_rots_quat[:,track_id]
            learned_trajectory_w2cams2[:,:3,:3] = quaternion_to_matrix(rot_quat)
            learned_trajectory_w2cams2[:,:3,3] = trans#[track_id] 
            learned_trajectory_cams2w = torch.linalg.inv(learned_trajectory_w2cams2)
            learned_tracklets[f'obj_tool{track_id+1}'] = {'trajectory_cams2w':learned_trajectory_cams2w}
        # self.cotrackerpnp_trajectory_cams2w = tracklets[f'obj_tool{i+1}']['trajectory_cams2w'].float().cuda()# 
        import os
        os.makedirs(saved_dir,exist_ok=True)
        saved_learned_tracklets_path = os.path.join(saved_dir,f'all_obj_tools_learned.pt')
        torch.save(learned_tracklets,saved_learned_tracklets_path)

    def save_state_dict(self, 
                        is_final = False,
                        ):
        state_dict = dict()
        if self.opt_track:
            state_dict['params'] = self.state_dict()
            if not is_final:
                state_dict['optimizer'] = self.optimizer.state_dict()
        return state_dict
