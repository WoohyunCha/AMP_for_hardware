import os
import glob
import json
import logging
from lxml import etree

import torch
import numpy as np
from pybullet_utils import transformations

from rsl_rl.utils import utils
from rsl_rl.datasets import pose3d
from rsl_rl.datasets import motion_util
from isaacgym.torch_utils import *
from scipy.signal import savgol_filter
from torch import optim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'mocap_motions', 'AMP_trajectories')
MJCF_file = '/home/cha/isaac_ws/AMP_for_hardware/resources/robots/tocabi/xml/dyros_tocabi_nomesh.xml'
FOOT_NAME = ['L_AnkleRoll_Joint', 'R_AnkleRoll_Joint']



class AMPLoader:
    FRAME_TIME = 0
    MODEL_DOF = 33
    REFERENCE_START_INDEX = None
    REFERENCE_END_INDEX = None


    POS_SIZE = 3
    JOINT_POS_SIZE = 12
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 12
    ROOT_POS_SIZE = 1 # base height 
    ROOT_ROT_SIZE = 4 # projected gravity

    OBSERVATION_DIM = ROOT_POS_SIZE + ROOT_ROT_SIZE + LINEAR_VEL_SIZE + ANGULAR_VEL_SIZE\
         + JOINT_POS_SIZE + JOINT_VEL_SIZE + 2*(POS_SIZE)

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + ROOT_POS_SIZE

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROOT_ROT_SIZE

    LINEAR_VEL_START_IDX = ROOT_ROT_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE

    JOINT_POSE_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    FOOT_POS_START_IDX = JOINT_VEL_END_IDX
    FOOT_POS_END_IDX = FOOT_POS_START_IDX + 2*POS_SIZE


    HZ = None
    # # REFERENCE_START_INDEX = [int(5.6*HZ), int(4.4*HZ)]
    # # REFERENCE_END_INDEX = [int(7.4005*HZ), int(5.6005*HZ)]
    # # REFERENCE_START_INDEX = int(2.*HZ)
    # # REFERENCE_END_INDEX = int(5.6005*HZ)
    # REFERENCE_START_INDEX = int(2.0*HZ)
    # REFERENCE_END_INDEX = int(5.6005*HZ)


    #CMU
    # HZ = 120
    # REFERENCE_START_INDEX = int(0.*HZ)
    # REFERENCE_END_INDEX = int(3.0*HZ)



    def __init__(
            self,
            device,
            time_between_frames,
            preload_transitions=False,
            num_preload_transitions=1000000,
            reference_dict={}, play=False, iterations=3000,
            target_model_file=MJCF_file
            ):
        """Expert dataset provides AMP observations from Dog mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []
        for i, (jsonf, info) in enumerate(reference_dict.items()):
            motion_file = jsonf
            model_file = info['xml']
            hz = info['hz']
            reference_start_index = int(hz * info['start_time'])
            reference_end_index = int(hz * info['end_time'])
            AMPLoader.MODEL_DOF = info['model_dof']
            AMPLoader.REFERENCE_START_INDEX = int(reference_start_index)
            AMPLoader.REFERENCE_END_INDEX = int(reference_end_index)
            AMPLoader.HZ = hz
            motion_weight = info['weight']
            
            print(motion_file)
            self.trajectory_names.append(motion_file.split('.')[0])
            with open(motion_file, "r") as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"]) # motion data in json form.
                if (model_file == target_model_file or model_file == ''): 
                    print("No retargetting motion")
                    processed_data_full, info = self.process_data(motion_data, i)
                    processed_data_joints = processed_data_full[:, self.JOINT_POSE_START_IDX:self.JOINT_VEL_END_IDX]

                else:
                    print("RETARGET REFERENCE MOTIONS")
                    print("Target file : ", target_model_file)
                    print("Source file : ", info['xml'])
                    motion_retarget = self.MotionRetarget(source_model_path=model_file, target_model_path=target_model_file, device=self.device) # model_file is the source
                    processed_data_full, _ = motion_retarget.retarget(motion_data, i, play, iterations)
                    processed_data_full = processed_data_full.to(torch.float32)
                    processed_data_joints = processed_data_full[:, self.JOINT_POSE_START_IDX:self.JOINT_VEL_END_IDX]

                self.trajectories.append( # Only joint space
                    processed_data_joints
                    )
                self.trajectories_full.append(
                    processed_data_full
                )

                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(
                    # float(motion_json["MotionWeight"]))
                    float(motion_weight))
                frame_duration = float(motion_data[1,AMPLoader.FRAME_TIME] - motion_data[0,AMPLoader.FRAME_TIME])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (reference_end_index-reference_start_index - 1) * frame_duration
                # traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                # self.trajectory_num_frames.append(float(motion_data.shape[0]))
                self.trajectory_num_frames.append(float(reference_end_index - reference_start_index))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")
            print(f"Size of Reference Observation : {self.observation_dim}")
        
        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        self.preload_transitions = preload_transitions

        if self.preload_transitions:
            print(f'Preloading {num_preload_transitions} transitions')
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times) # error
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print(f'Finished preloading')


        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(
            self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights,
            replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(
            0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, dtype=torch.float32, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, dtype=torch.float32, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32, requires_grad=False).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        all_frame_pos_starts = torch.zeros(len(traj_idxs), AMPLoader.ROOT_POS_SIZE, dtype=torch.float32, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), AMPLoader.ROOT_POS_SIZE, dtype=torch.float32, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), AMPLoader.ROOT_ROT_SIZE, dtype=torch.float32, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), AMPLoader.ROOT_ROT_SIZE, dtype=torch.float32, device=self.device)
        all_frame_linvel_starts = torch.zeros(len(traj_idxs), AMPLoader.LINEAR_VEL_SIZE, dtype=torch.float32, device=self.device)
        all_frame_linvel_ends = torch.zeros(len(traj_idxs), AMPLoader.LINEAR_VEL_SIZE, dtype=torch.float32, device=self.device)
        all_frame_angvel_starts = torch.zeros(len(traj_idxs), AMPLoader.ANGULAR_VEL_SIZE, dtype=torch.float32, device=self.device)
        all_frame_angvel_ends = torch.zeros(len(traj_idxs), AMPLoader.ANGULAR_VEL_SIZE, dtype=torch.float32, device=self.device)
        all_frame_amp_starts = torch.zeros(len(traj_idxs), AMPLoader.JOINT_VEL_END_IDX - AMPLoader.JOINT_POSE_START_IDX, dtype=torch.float32, device=self.device)
        all_frame_amp_ends = torch.zeros(len(traj_idxs),  AMPLoader.JOINT_VEL_END_IDX - AMPLoader.JOINT_POSE_START_IDX, dtype=torch.float32, device=self.device)
        all_frame_foot_pos_starts = torch.zeros(len(traj_idxs), AMPLoader.FOOT_POS_END_IDX - AMPLoader.FOOT_POS_START_IDX, dtype=torch.float32, device=self.device)
        all_frame_foot_pos_ends = torch.zeros(len(traj_idxs), AMPLoader.FOOT_POS_END_IDX - AMPLoader.FOOT_POS_START_IDX, dtype=torch.float32, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = AMPLoader.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = AMPLoader.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = AMPLoader.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = AMPLoader.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_linvel_starts[traj_mask] = AMPLoader.get_linear_vel_batch(trajectory[idx_high[traj_mask]])
            all_frame_linvel_ends[traj_mask] = AMPLoader.get_linear_vel_batch(trajectory[idx_low[traj_mask]])
            all_frame_angvel_starts[traj_mask] = AMPLoader.get_angular_vel_batch(trajectory[idx_high[traj_mask]])
            all_frame_angvel_ends[traj_mask] = AMPLoader.get_angular_vel_batch(trajectory[idx_low[traj_mask]])
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][:, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_VEL_END_IDX]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][:, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_VEL_END_IDX]
            all_frame_foot_pos_starts[traj_mask] = AMPLoader.get_foot_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_foot_pos_ends[traj_mask] = AMPLoader.get_foot_pos_batch(trajectory[idx_high[traj_mask]])

        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32, requires_grad=False).unsqueeze(-1)

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend) 
        rot_blend = self.slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        lin_blend = self.slerp(all_frame_linvel_starts, all_frame_linvel_ends, blend)
        ang_blend = self.slerp(all_frame_angvel_starts, all_frame_angvel_ends, blend)
        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        foot_pos_blend = self.slerp(all_frame_foot_pos_starts, all_frame_foot_pos_ends, blend)

        ret = torch.cat([pos_blend, rot_blend, lin_blend, ang_blend, amp_blend, foot_pos_blend], dim=-1) 
        # ret = torch.cat([pos_blend, rot_blend, amp_blend, foot_pos_blend], dim=-1) 
        # ret = torch.cat([rot_blend, amp_blend, foot_pos_blend], dim=-1) 
        return ret

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(
                self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        root_pos0, root_pos1 = AMPLoader.get_root_pos(frame0), AMPLoader.get_root_pos(frame1)
        root_rot0, root_rot1 = AMPLoader.get_root_rot(frame0), AMPLoader.get_root_rot(frame1)
        linear_vel_0, linear_vel_1 = AMPLoader.get_linear_vel(frame0), AMPLoader.get_linear_vel(frame1)
        angular_vel_0, angular_vel_1 = AMPLoader.get_angular_vel(frame0), AMPLoader.get_angular_vel(frame1)
        joints0, joints1 = AMPLoader.get_joint_pose(frame0), AMPLoader.get_joint_pose(frame1)
        joint_vel_0, joint_vel_1 = AMPLoader.get_joint_vel(frame0), AMPLoader.get_joint_vel(frame1)
        foot_pos0, foot_pos1 = AMPLoader.get_foot_pos(frame0), AMPLoader.get_foot_pos(frame1)
        
        blend_root_pos = self.slerp(root_pos0, root_pos1, blend) 
        blend_root_rot = self.slerp(root_rot0, root_rot1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)
        blend_foot_pos = self.slerp(foot_pos0, foot_pos1, blend)

        ret = torch.cat([
            blend_root_pos, blend_root_rot, blend_linear_vel, blend_angular_vel, blend_joints, blend_joints_vel,
            blend_foot_pos
            ], dim=-1)
        # ret = torch.cat([ 
        #     blend_root_pos, blend_root_rot, blend_joints, blend_joints_vel,
        #     blend_foot_pos
        #     ], dim=-1)
        # ret = torch.cat([
        #     blend_root_rot, blend_joints, blend_joints_vel,
        #     blend_foot_pos
        #     ], dim=-1)

        return ret

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(
                    self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[idxs, :]
                s_next = self.preloaded_s_next[idxs, :]

            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    s.append(self.get_full_frame_at_time(traj_idx, frame_time))
                    s_next.append(
                        self.get_full_frame_at_time(
                            traj_idx, frame_time + self.time_between_frames))
                
                s = torch.vstack(s)
                s_next = torch.vstack(s_next)
            yield s.to(torch.float32), s_next.to(torch.float32)

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.OBSERVATION_DIM

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_root_pos(pose):
        return pose[AMPLoader.ROOT_POS_START_IDX:AMPLoader.ROOT_POS_END_IDX]

    def get_root_pos_batch(poses):
        return poses[:, AMPLoader.ROOT_POS_START_IDX:AMPLoader.ROOT_POS_END_IDX]

    def get_root_rot(pose):
        return pose[AMPLoader.ROOT_ROT_START_IDX:AMPLoader.ROOT_ROT_END_IDX]

    def get_root_rot_batch(poses):
        return poses[:, AMPLoader.ROOT_ROT_START_IDX:AMPLoader.ROOT_ROT_END_IDX]

    def get_linear_vel(pose):
        return pose[AMPLoader.LINEAR_VEL_START_IDX:AMPLoader.LINEAR_VEL_END_IDX]
    
    def get_linear_vel_batch(poses):
        return poses[:, AMPLoader.LINEAR_VEL_START_IDX:AMPLoader.LINEAR_VEL_END_IDX]

    def get_angular_vel(pose):
        return pose[AMPLoader.ANGULAR_VEL_START_IDX:AMPLoader.ANGULAR_VEL_END_IDX]  

    def get_angular_vel_batch(poses):
        return poses[:, AMPLoader.ANGULAR_VEL_START_IDX:AMPLoader.ANGULAR_VEL_END_IDX]
    
    def get_joint_pose(pose):
        return pose[AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_POSE_END_IDX]
  
    def get_joint_vel(pose):
        return pose[AMPLoader.JOINT_VEL_START_IDX:AMPLoader.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, AMPLoader.JOINT_VEL_START_IDX:AMPLoader.JOINT_VEL_END_IDX]
          
    def get_foot_pos(pose):
        return pose[AMPLoader.FOOT_POS_START_IDX:AMPLoader.FOOT_POS_END_IDX]

    def get_foot_pos_batch(poses):
        return poses[:, AMPLoader.FOOT_POS_START_IDX:AMPLoader.FOOT_POS_END_IDX]

    @staticmethod
    def process_data(traj: np.ndarray, index: int):# Process raw data from Mujoco
        traj = denoise(traj, window_length=int(0.2*AMPLoader.HZ), polyorder=2)
        time = traj[AMPLoader.REFERENCE_START_INDEX:AMPLoader.REFERENCE_END_INDEX,0]
        pelv_pos = traj[AMPLoader.REFERENCE_START_INDEX:AMPLoader.REFERENCE_END_INDEX, 1:4]
        pelv_rpy = traj[AMPLoader.REFERENCE_START_INDEX:AMPLoader.REFERENCE_END_INDEX, 4:7] # roll, pitch, yaw order
        pelv_vel = traj[AMPLoader.REFERENCE_START_INDEX:AMPLoader.REFERENCE_END_INDEX, 7:13]
        q_pos = traj[AMPLoader.REFERENCE_START_INDEX:AMPLoader.REFERENCE_END_INDEX, 13:13+AMPLoader.JOINT_POS_SIZE]
        start = 13+AMPLoader.MODEL_DOF
        end = start + AMPLoader.JOINT_VEL_SIZE
        q_vel = traj[AMPLoader.REFERENCE_START_INDEX:AMPLoader.REFERENCE_END_INDEX, start:end]
        start += AMPLoader.MODEL_DOF
        end = start + 3
        L_foot_pos = traj[AMPLoader.REFERENCE_START_INDEX:AMPLoader.REFERENCE_END_INDEX, start:end]
        start = end
        end = start + 3
        L_foot_rpy = traj[AMPLoader.REFERENCE_START_INDEX:AMPLoader.REFERENCE_END_INDEX, start:end]
        start = end
        end = start + 3
        R_foot_pos = traj[AMPLoader.REFERENCE_START_INDEX:AMPLoader.REFERENCE_END_INDEX, start:end]
        start=end
        end = start + 3
        R_foot_rpy = traj[AMPLoader.REFERENCE_START_INDEX:AMPLoader.REFERENCE_END_INDEX, start:end]
        assert end == 1+3+3+6+AMPLoader.MODEL_DOF+AMPLoader.MODEL_DOF+3+3+3+3, f"process data, shape mismatch {end}"

        # torch versions of data
        pelv_pos_torch = to_torch(pelv_pos, dtype=torch.float32)
        pelv_vel_torch = to_torch(pelv_vel, dtype=torch.float32)
        q_pos_torch = to_torch(q_pos, dtype=torch.float32)
        q_vel_torch = to_torch(q_vel, dtype=torch.float32)
        L_foot_pos_torch = to_torch(L_foot_pos, dtype=torch.float32)
        L_foot_rpy_torch = to_torch(L_foot_rpy, dtype=torch.float32)
        R_foot_pos_torch = to_torch(R_foot_pos, dtype=torch.float32)
        R_foot_rpy_torch = to_torch(R_foot_rpy, dtype=torch.float32)

        # Process raw data
        pelvis_yaw =pelv_rpy[:, 2]
        pelvis_pitch = pelv_rpy[:, 1]
        pelvis_roll = pelv_rpy[:, 0]
        pelvis_quat = quat_from_euler_xyz(to_torch(pelvis_roll, dtype=torch.float32), to_torch(pelvis_pitch, dtype=torch.float32), to_torch(pelvis_yaw, dtype=torch.float32)) # tensor
        base_pos_global_torch = pelv_pos_torch

        # Create AMP observation
        base_height = base_pos_global_torch[:, 2]
        assert (~torch.isfinite(base_height)).sum() == 0, "Found non finite element"
        assert (~torch.isfinite(pelvis_quat)).sum() == 0, "Found non finite element 0"
        # gravity_vector = to_torch(get_axis_params(-1., 2), device=self.device).repeat((pelvis_yaw.shape[0], 1))
        # projected_gravity = quat_rotate_inverse(pelvis_quat, gravity_vector)
        base_lin_vel = quat_rotate_inverse(pelvis_quat, pelv_vel_torch[:, :3]) 
        assert (~torch.isfinite(base_lin_vel)).sum() == 0, "Found non finite element 1"
        base_ang_vel = quat_rotate_inverse(pelvis_quat, pelv_vel_torch[:, 3:])
        assert (~torch.isfinite(base_ang_vel)).sum() == 0, "Found non finite element 2"
        L_foot_pos_base_torch = quat_rotate_inverse(pelvis_quat, L_foot_pos_torch - base_pos_global_torch)
        assert (~torch.isfinite(L_foot_pos_base_torch)).sum() == 0, "Found non finite element 3"
        R_foot_pos_base_torch = quat_rotate_inverse(pelvis_quat, R_foot_pos_torch - base_pos_global_torch) 
        assert (~torch.isfinite(R_foot_pos_base_torch)).sum() == 0, "Found non finite element 4"
        L_foot_quat_global_torch = quat_from_euler_xyz(L_foot_rpy_torch[:, 0], L_foot_rpy_torch[:, 1], L_foot_rpy_torch[:, 2])
        assert (~torch.isfinite(L_foot_quat_global_torch)).sum() == 0, "Found non finite element 5"
        R_foot_quat_global_torch = quat_from_euler_xyz(R_foot_rpy_torch[:, 0], R_foot_rpy_torch[:, 1], R_foot_rpy_torch[:, 2])
        assert (~torch.isfinite(R_foot_quat_global_torch)).sum() == 0, "Found non finite element 6"
        L_foot_quat_base_torch = quat_mul(quat_conjugate(pelvis_quat), L_foot_quat_global_torch)
        assert (~torch.isfinite(L_foot_quat_base_torch)).sum() == 0, "Found non finite element 7"
        R_foot_quat_base_torch = quat_mul(quat_conjugate(pelvis_quat), R_foot_quat_global_torch)
        assert (~torch.isfinite(R_foot_quat_base_torch)).sum() == 0, "Found non finite element 8"
        ret = torch.concat((base_height.unsqueeze(-1), pelvis_quat, base_lin_vel, base_ang_vel, q_pos_torch, q_vel_torch, L_foot_pos_base_torch, R_foot_pos_base_torch), dim=-1) 
        # ret = torch.concat((base_height.unsqueeze(-1), pelvis_quat, q_pos_torch, q_vel_torch, L_foot_pos_base_torch, R_foot_pos_base_torch), dim=-1) 
        # ret = torch.concat((pelvis_quat, q_pos_torch, q_vel_torch, L_foot_pos_base_torch, R_foot_pos_base_torch), dim=-1) 
        info = torch.concat((base_height.unsqueeze(-1), base_lin_vel, base_ang_vel, L_foot_quat_base_torch, R_foot_quat_base_torch), dim=-1)
        with open(os.path.join(PROCESSED_DATA_DIR, 'processed_data_'+str(index)+'.txt'), "w") as file:
            for line in ret.cpu().numpy().tolist():
                file.write(' '.join(map(str, line)) + '\n')
            print(f"Processed data {index} written to txt file at "+PROCESSED_DATA_DIR)
        return ret, info
    
    class MotionRetarget():

        JOINT_MAPPING = {
            'virtual_joint': 0,
            'L_HipYaw_Joint': 7, 
            'L_HipRoll_Joint': 8,
            'L_HipPitch_Joint': 9,
            'L_Knee_Joint': 10,
            'L_AnklePitch_Joint': 11,
            'L_AnkleRoll_Joint': 12,

            'R_HipYaw_Joint': 13, 
            'R_HipRoll_Joint': 14,
            'R_HipPitch_Joint': 15,
            'R_Knee_Joint': 16,
            'R_AnklePitch_Joint': 17,
            'R_AnkleRoll_Joint': 18,
        }

        def __init__(self, source_model_path: str, target_model_path: str, device):
            source = source_model_path
            target = target_model_path
            source_bodies, source_joints, source_edges = self.process_model(source)
            target_bodies, target_joints, target_edges = self.process_model(target)
            self.root_ratio = target_bodies['base_link']['info']['position'][-1] / source_bodies['base_link']['info']['position'][-1]
            assert self.root_ratio < 10 and self.root_ratio > 0.1, f"Root ratio is abnormally small, {self.root_ratio}"
            self.source = {
                'bodies': source_bodies,
                'joints': source_joints,
                'edges': source_edges
            }
            self.target = {
                'bodies': target_bodies,
                'joints': target_joints,
                'edges': target_edges
            }
            self.source_model_path = source_model_path
            self.target_model_path = target_model_path
            # print("SOURCE MODEL")
            # self.print_model(self.source)
            # print("TARGET MODEL")
            # self.print_model(self.target)
            self.device = device

        def process_model(self, model_path: str):
            tree = etree.parse(model_path)
            root = tree.getroot()
            joints = {}
            bodies = {}
            edges = {}

        # Collect body and joint data
            for body in root.findall('.//body'):
                body_name = body.get('name')
                pos = np.array([float(num) for num in body.get('pos', '0 0 0').split()], dtype=float) # default value is 0, 0, 0
                quat = np.array([float(num) for num in body.get('quat', '1 0 0 0').split()], dtype=float) # default value is 0, 0, 0
                if body.find('inertial') is not None: # not a dummy
                    inertial = body.find('inertial')
                    mass = np.array([float(num) for num in inertial.get('mass', '0.').split()], dtype=float) # default value is 0, 0, 0
                    com = np.array([float(num) for num in inertial.get('pos', '0 0 0').split()], dtype=float) # default value is 0, 0, 0
                    inertia = np.array([float(num) for num in inertial.get('fullinertia', '0 0 0 0 0 0').split()], dtype=float) # default value is 0, 0, 0
                else:
                    mass = None
                    com = None
                    inertia = None

                body_info = {
                    'position': pos,
                    'quat': quat,
                    'mass': mass,
                    'inertia': inertia,
                    'com': com
                }

                parent = body.getparent()
                parent_name = parent.get('name')
                if parent_name is None:
                    parent_name = 'world'
                
                if body.find('joint') is not None:
                    for i, joint in enumerate(body.findall('joint')):
                        joint_name = joint.get('name')
                        joint_pos = np.array([float(num) for num in joint.get('pos', '0 0 0').split()], dtype=float)
                        joint_axis = np.array([float(num) for num in joint.get('axis', '0 0 1').split()], dtype=float)
                        joint_quat = np.array([float(num) for num in joint.get('quat', '1 0 0 0').split()], dtype=float)
                        joint_type = joint.get('type', 'hinge')
                        joint_range = np.array([float(num) for num in joint.get('range', '-3.14 3.14').split()], dtype=float)
                        joint_info = {
                            'parent' : parent_name,
                            'child' : body_name,
                            'position' : joint_pos,
                            'axis' : joint_axis,
                            'type' : joint_type,
                            'range' : joint_range,
                            'quat' : joint_quat
                        }
                        joints[joint_name] = joint_info
                        # process link length between joint_name and parent joint
                        edges[joint_name] = np.linalg.norm(body_info['position'], ord=2, axis=-1)
                else: # fixed link
                    joint_name = None             
                bodies[body_name] = {'parent': parent_name, 'joint': joint_name, 'info' : body_info}    
                m = {'bodies': bodies, 'joints': joints}

            return bodies, joints, edges

        def print_model(self, model):
            bodies, joints, edges = model['bodies'], model['joints'], model['edges']
            print("PRINTING MODEL")
            print("---------------INFO----------------")
            for body_name, val in bodies.items():
                print("Link name : ", body_name)
                print(val)
                print("Joint name : ", val['joint'])
                print(joints[val['joint']]) if val['joint'] is not None else print("None")
                print("----------------------")
            print("-------------Link length-----------")
            for joint_name, val in edges.items():
                if joints[joint_name]['child'] != "base_link":
                    print(f"{joint_name} and {bodies[joints[joint_name]['parent']]['joint']} : {val}")

        def retarget_tensor(self, reference: torch.Tensor, iterations :int,  play=False) -> torch.Tensor: # outputs retargetted reference
            length_ratios = {}
            reference_length = reference.shape[0]
            observation_dim = reference.shape[-1]
            for key, val in self.target['edges'].items():
                if key in self.JOINT_MAPPING:
                    if self.source['edges'][key] == 0:
                        length_ratios[key] = 0.
                    else:
                        length_ratios[key] = val/self.source['edges'][key]
            source_global_joint_pos = {}
            source_global_joint_quat = {}
            target_global_joint_pos = {}
            target_local_joint_pos = {}
            target_global_joint_quat = {}
            target_local_joint_quat = {}
            relative_position_vectors = {}
            source_bodies, source_joints= self.source['bodies'], self.source['joints']
            target_bodies, target_joints = self.target['bodies'], self.target['joints']

            start = 0
            end = start+AMPLoader.ROOT_POS_SIZE
            # source_global_root_pos = torch.cat((torch.zeros_like(reference[:, start:end]), torch.zeros_like(reference[:, start:end]), reference[:, start:end]), dim=-1)
            source_global_root_pos = torch.cat((torch.zeros_like(reference[:, 0:1]), torch.zeros_like(reference[:, 0:1]), torch.zeros_like(reference[:, 0:1])), dim=-1)
            start = end
            end = start+AMPLoader.ROOT_ROT_SIZE
            source_global_root_quat = reference[:, start:end]
            start = end
            end = start+AMPLoader.LINEAR_VEL_SIZE
            source_local_base_lin_vel = reference[:, start:end]
            start = end
            end = start+AMPLoader.ANGULAR_VEL_SIZE
            source_local_base_ang_vel = reference[:, start:end]
            start = end
            end = start+AMPLoader.JOINT_POS_SIZE
            source_qpos = reference[:, start:end]
            start = end
            end = start+AMPLoader.JOINT_VEL_SIZE
            source_qvel = reference[:, start:end]
            start = end
            end = start+AMPLoader.POS_SIZE
            source_local_Lfoot_pos = reference[:, start:end]
            start = end
            end = start+AMPLoader.POS_SIZE
            source_local_Rfoot_pos = reference[:, start:end]
            assert end == reference.shape[-1], f"Retarget, observation shape do not match. Must be {reference.shape[-1]} but is {end}"
            
            source_q = torch.concat((source_global_root_pos, source_global_root_quat, reference[:, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_POSE_END_IDX]), dim=-1)
            assert source_q.shape == (reference_length, 7+12)
            source_global_joint_pos, source_global_joint_quat, source_local_joint_pos, source_local_joint_quat = self.forward_kinematics(self.source, source_q, self.JOINT_MAPPING)
            for body_name, body in source_bodies.items():

                parent_name = body['parent']
                joint_name = body['joint']
                body_info = body['info']

                if joint_name in self.JOINT_MAPPING and body_name != "base_link":
                    if 'HipYaw_Joint' in joint_name:
                        relative_position_vectors[joint_name] = to_torch(length_ratios[joint_name]).tile((reference_length, 1)) * \
                            (source_global_joint_pos[joint_name] - source_global_joint_pos['virtual_joint']) # scaled relative position vector from parent joint to current joint in global frame                    
                    else:
                        relative_position_vectors[joint_name] = to_torch(length_ratios[joint_name]).tile((reference_length, 1)) * \
                            (source_global_joint_pos[joint_name] - source_global_joint_pos[source_bodies[parent_name]['joint']]) # scaled relative position vector from parent joint to current joint in global frame

            # 3. Starting from base link, sum up the relative position vectors and transform back to base frame coordinates              
            joint_pos_global_retarget = torch.zeros((reference_length, 3 * 13), dtype=torch.float, device='cuda:0')
            joint_pos_local_retarget = torch.zeros((reference_length, 3 * 12), dtype=torch.float, device='cuda:0')
            joint_quat_global_retarget = torch.zeros((reference_length, 4*13), dtype=torch.float, device='cuda:0')
            joint_quat_local_retarget = torch.zeros((reference_length, 4*12), dtype=torch.float, device='cuda:0')
            for body_name, body in source_bodies.items():
                parent_name = body['parent']
                joint_name = body['joint']
                body_info = body['info']
                base_pos = source_global_joint_pos[source_bodies['base_link']['joint']]
                base_quat = source_global_joint_quat[source_bodies['base_link']['joint']]

                if body_name == "base_link":
                    target_global_joint_pos[joint_name] = base_pos
                    target_global_joint_quat[joint_name] = base_quat
                    joint_pos_global_retarget[:, 0:3] = base_pos
                elif joint_name in self.JOINT_MAPPING and body_name != "base_link":
                    parent_joint_name = source_bodies[parent_name]['joint']
                    if 'HipYaw_Joint' in joint_name:
                        parent_joint_name = 'virtual_joint'
                    target_global_joint_pos[joint_name] = target_global_joint_pos[parent_joint_name] + relative_position_vectors[joint_name]
                    target_global_joint_quat[joint_name] = source_global_joint_quat[joint_name]
                    target_local_joint_pos[joint_name] = quat_rotate_inverse(base_quat, (target_global_joint_pos[joint_name] - base_pos)) # relative position of joint from base in base frame coordinates
                    target_local_joint_quat[joint_name] = quat_mul(quat_conjugate(base_quat), target_global_joint_quat[joint_name]) # quat of joint frame in base frame coordinates. Only the foot orientation will be a constraint for the IK
                    joint_pos_global_retarget[:, 3*(self.JOINT_MAPPING[joint_name]-6):3*(self.JOINT_MAPPING[joint_name]-5)] = target_global_joint_pos[joint_name]
                    joint_pos_local_retarget[:, 3*(self.JOINT_MAPPING[joint_name]-7):3*(self.JOINT_MAPPING[joint_name]-6)] = target_local_joint_pos[joint_name]
                    joint_quat_global_retarget[:, 4*(self.JOINT_MAPPING[joint_name]-6):4*(self.JOINT_MAPPING[joint_name]-5)] = target_global_joint_quat[joint_name]
                    joint_quat_local_retarget[:, 4*(self.JOINT_MAPPING[joint_name]-7):4*(self.JOINT_MAPPING[joint_name]-6)] = target_local_joint_quat[joint_name]

            # # 4. Solve numerical IK from joint positions, foot positions and orientation
            weight_joint_pos, weight_joint_pose, weight_joint_vel = 1., .5, .1
            # weight_joint_pos, weight_foot_quat, weight_joint_pose, weight_hip_pose, weight_joint_vel = 1.2, 0.1, .1, 0.1, .1 For CMU refernece
            weight_norm = (weight_joint_pos+weight_joint_pose)
            weight_joint_pos /= weight_norm
            weight_joint_pose/= weight_norm
            weight_joint_vel /= weight_norm

            def cost_function(x):
                # x = [q for each joints] -> shape of 12
                # Initialize x as [0, reference[:, ROOT_ROT_START_IDX:ROOT_ROT_END_IDX], reference[:, JOINT_POSE_START_IDX:JOINT_POSE_END_IDX]]
                # joint_pos_global_retarget = [base_pos, joint_pos] in global frame coordinates
                # joint_pos_local_retarget = [joint_pos] in base local frame coordinates

                temp = torch.zeros((x.shape[0], 7), requires_grad=False, device='cuda:0')
                temp[:, -1] = 1.
                _, x_global_joint_quat, x_local_joint_pos, x_local_joint_quat = self.forward_kinematics(self.target, torch.cat((temp, x), dim=-1), self.JOINT_MAPPING)
                x_local_joint_pos_tensor = torch.zeros_like(joint_pos_local_retarget)
                x_local_joint_quat_tensor = torch.zeros_like(joint_quat_local_retarget)
                for joint_name, joint_index in self.JOINT_MAPPING.items():
                    if key == 'virtual_joint':
                        pass
                    else:
                        index = joint_index-7
                        x_local_joint_pos_tensor[:, 3*index:3*(index+1)] = x_local_joint_pos[joint_name]
                        x_local_joint_quat_tensor[:, 4*index:4*(index+1)] = x_local_joint_quat[joint_name]
                assert x_local_joint_pos_tensor.shape == (reference_length, 36), f"local joint pos tensor shape is {x_local_joint_pos_tensor.shape}"

                # mask_ref = [i+JOINT_POSE_START_IDX for i in mask]
                # joint_pose_cost = torch.nn.MSELoss()(x[:, mask], TOCABI_INIT_POS_TORCH[:, mask]) # For CMU reference
                joint_pose_cost = torch.nn.MSELoss()(x, source_qpos)
                joint_pos_cost = torch.nn.MSELoss()(x_local_joint_pos_tensor, joint_pos_local_retarget)
                joint_vel_cost = torch.nn.MSELoss()(x[1:], x[:-1])
    # 
                L_ankle_roll_joint = FOOT_NAME[0]
                R_ankle_roll_joint = FOOT_NAME[1]

                x_Lfoot_quat = x_global_joint_quat[L_ankle_roll_joint]
                x_Rfoot_quat = x_global_joint_quat[R_ankle_roll_joint]
                zero_quat = torch.tensor([0, 0, 0, 1], dtype=torch.float, device='cuda:0', requires_grad=False).tile(reference_length,1)
                # foot_quat_cost = (distance_between_quats(zero_quat, x_Lfoot_quat, neglect_axis=2).pow(2) \
                #                   + distance_between_quats(zero_quat, x_Rfoot_quat, neglect_axis=2).pow(2)).mean()
                cosine_loss = torch.nn.CosineSimilarity(dim=-1)

                total_cost = weight_joint_pos*joint_pos_cost + weight_joint_pose*joint_pose_cost + weight_joint_vel*joint_vel_cost
                return torch.sum(total_cost), joint_pos_cost, joint_pose_cost, (1 - cosine_loss(zero_quat, x_Lfoot_quat)).mean(), (1 - cosine_loss(zero_quat, x_Rfoot_quat)).mean()
            
            n_iterations = iterations
            if play:
                n_iterations = 0
            # q_opt = torch.zeros((reference_length, 12), dtype=torch.float32, requires_grad=True, device='cuda:0')
            # q_opt = reference[:, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_POSE_END_IDX].clone().detach()
            TOCABI_INIT_POS_TORCH = torch.tensor([0.0, 0.0, -0.28, 0.6, -0.32, 0.0, 0.0, 0.0, -0.28, 0.6, -0.32, 0.], dtype=torch.float32, device=self.device)
            q_opt = (TOCABI_INIT_POS_TORCH.tile((reference_length, 1))).clone().detach()
            q_opt += 0.01*torch.rand_like(q_opt)
            q_opt = q_opt.requires_grad_(True)
            assert q_opt.shape == (reference_length, 12)
            optimizer = optim.Adam([q_opt], lr=1e-3)

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)
            print("SOLVING IK...")
            for i in range(n_iterations):
                optimizer.zero_grad()
                cost = cost_function(q_opt)[0]
                cost.backward()
                torch.nn.utils.clip_grad_norm_(q_opt, max_norm=1.0)
                optimizer.step()
                scheduler.step(cost)
                if i%100 == 0:
                    print(f"Iteration {i}, Cost: {cost.item()}")

                # Optionally, you can add some stopping criterion based on the change in cost or other conditions
                if optimizer.param_groups[0]['lr'] < 1e-6:
                    print("Stopping criterion met")
                    break
                assert ~torch.any(~torch.isfinite(q_opt)), f"Cannot solve IK! Cost : {cost_function(q_opt)}"
            print("IK SOLVED!")

            temp = torch.zeros((q_opt.shape[0], 7), requires_grad=False, device='cuda:0')
            temp[:, :3] = to_torch(target_bodies['base_link']['info']['position'], device='cuda:0').tile((q_opt.shape[0], 1))
            temp[:, -1] = 1.        
            retarget_global_joint_pos, retarget_global_joint_quat, retarget_local_joint_pos, retarget_local_joint_quat = self.forward_kinematics(self.target, torch.cat((temp, q_opt), dim=-1), self.JOINT_MAPPING)
            ### FROM HEREON, CONSTRUCT THE RETARGET TRAJECTORY!!!###
            q_opt_prev, q_vel_opt = torch.zeros_like(q_opt), torch.zeros_like(q_opt)

            q_opt_prev[1:] = q_opt[:-1]
            q_vel_opt[1:] = (q_opt - q_opt_prev)[1:] * AMPLoader.HZ
            L_foot_pos_opt = retarget_local_joint_pos[FOOT_NAME[0]]
            R_foot_pos_opt = retarget_local_joint_pos[FOOT_NAME[1]]
            assert q_vel_opt.shape[0] == L_foot_pos_opt.shape[0]
            # time = torch.tensor([i for i in range(q_vel_opt.shape[0])], dtype=torch.float32, device='cuda:0').view(-1,1)
            # retarget_reference = torch.cat((retarget_global_joint_quat['virtual_joint'], q_opt, q_vel_opt, L_foot_pos_opt, R_foot_pos_opt), dim=-1)
            retarget_reference = torch.cat((self.root_ratio*reference[:, 0:1],retarget_global_joint_quat['virtual_joint'], self.root_ratio*source_local_base_lin_vel, source_local_base_ang_vel, q_opt, q_vel_opt, L_foot_pos_opt, R_foot_pos_opt), dim=-1)

            target_reference = torch.zeros((q_opt.shape[0], 3*13), requires_grad=False, device='cuda:0')
            for joint_name, _ in self.JOINT_MAPPING.items():
                target_reference[:, 3*(self.JOINT_MAPPING[joint_name]-6):3*(self.JOINT_MAPPING[joint_name]-5)] = retarget_global_joint_pos[joint_name]#target_global_joint_pos[joint_name] # retarget_global_joint_pos[joint_name]
            target_reference[:, 0:3] = retarget_global_joint_pos['virtual_joint'][:, 0:3]#target_global_joint_pos['virtual_joint'][:, 0:3] # retarget_global_joint_pos['virtual_joint'][:, 0:3]

            # return target_reference.detach()
            return retarget_reference.detach(), target_reference.detach()
        
        def retarget(self, source_motion_data, index, play=False, iterations=2000):
            source_reference, source_info = AMPLoader.process_data(source_motion_data, index)
            return self.retarget_tensor(source_reference, iterations, play)

        def forward_kinematics(self, model: dict, q: torch.Tensor, joint_map: dict, device = 'cuda:0', requires_grad = False):
            '''
            Computes the forward kinematics for a trajectory, given model = {bodies: dict, joints: dict}, q = torch.Tensor((traj_length, num_joints+7))
            joint_map is a dict that maps the joint names to their indices in q. For example, virtual joint = 0, hip yaw joint = 7, ...
            '''
            VIRTUAL_JOINT_DIM = 7 # pos and quat
            TRAJECTORY_LENGTH = q.shape[0]

            bodies, joints = model['bodies'], model['joints']
            num_joints = q.shape[-1]-VIRTUAL_JOINT_DIM # q includes virtual joint. The rotation part is in quat.
            # The position and orientation of the frame attached to the joint. This will be the body frame for the child link.
            global_joint_pos = {}
            global_body_pos = {}
            global_joint_rot = {}
            global_body_rot = {}
            local_body_pos = {}
            local_body_rot = {}
            local_joint_pos = {}
            local_joint_rot = {}

            global_base_rot = None

            for body_name, body in bodies.items():
                parent_name = body['parent']
                joint_name = body['joint']
                body_info = body['info']
                if joint_name is not None:
                    joint_info = joints[joint_name]

                if body_name == "base_link": # Let's assume there is no virtual joint offset from base link frame origin
                    virtual_joint_value = q[:, joint_map[joint_name]:joint_map[joint_name]+VIRTUAL_JOINT_DIM]
                    global_joint_pos[joint_name] = virtual_joint_value[:, 0:3]
                    global_joint_rot[joint_name] = virtual_joint_value[:, 3:VIRTUAL_JOINT_DIM]
                    global_body_pos[body_name] = global_joint_pos[joint_name]
                    global_body_rot[body_name] = global_joint_rot[joint_name]
                    global_base_pos = global_joint_pos[joint_name]
                    global_base_rot = global_joint_rot[joint_name]
                    local_joint_pos[joint_name] = torch.zeros_like(global_joint_pos[joint_name])
                    local_joint_rot[joint_name] = torch.zeros_like(global_joint_rot[joint_name])
                    local_body_pos[body_name] = torch.zeros_like(global_base_pos)
                    local_body_rot[body_name] = torch.zeros_like(global_base_rot)

                    local_joint_rot[joint_name][:, -1] = 1. # Quaternion zero.
                elif joint_name in joint_map:
                    q_ = q[:, joint_map[joint_name]]
                    body_offset = to_torch(body_info['position']).tile(TRAJECTORY_LENGTH, 1)
                    joint_offset = to_torch(joint_info['position']).tile(TRAJECTORY_LENGTH, 1)
                    body_rot_offset = to_torch(body_info['quat']).tile(TRAJECTORY_LENGTH, 1)[:, [1,2,3,0]]
                    joint_rot_offset = to_torch(joint_info['quat']).tile(TRAJECTORY_LENGTH, 1)[:, [1,2,3,0]]
                    global_body_rot[body_name] = quat_mul(quat_mul(global_body_rot[parent_name], quat_from_angle_axis(q_, to_torch(joint_info['axis']))), body_rot_offset)
                    global_body_pos[body_name] = global_body_pos[parent_name] + quat_rotate(global_body_rot[parent_name], body_offset)
                    global_joint_pos[joint_name] = global_body_pos[body_name] + quat_rotate(global_body_rot[body_name], joint_offset)
                    global_joint_rot[joint_name] = quat_mul(global_body_rot[body_name], joint_rot_offset)

                    local_body_pos[joint_name] = quat_rotate_inverse(global_base_rot, (global_body_pos[body_name] - global_base_pos))
                    local_body_rot[joint_name] = quat_mul(quat_conjugate(global_base_rot), global_body_rot[body_name])
                    local_joint_pos[joint_name] = quat_rotate_inverse(global_base_rot, (global_joint_pos[joint_name] - global_base_pos))
                    local_joint_rot[joint_name] = quat_mul(quat_conjugate(global_base_rot), global_joint_rot[joint_name])

                elif joint_name is None:
                    # body is welded to parent
                    body_offset = to_torch(body_info['position']).tile(TRAJECTORY_LENGTH, 1)
                    body_rot_offset = to_torch(body_info['quat']).tile(TRAJECTORY_LENGTH, 1)[:, [1,2,3,0]]
                    global_body_rot[body_name] = quat_mul(global_body_rot[parent_name], body_rot_offset)
                    global_body_pos[body_name] = global_body_pos[parent_name] + quat_rotate(global_body_rot[parent_name], body_offset)
                    local_body_pos[joint_name] = quat_rotate_inverse(global_base_rot, (global_body_pos[body_name] - global_base_pos))
                    local_body_rot[joint_name] = quat_mul(quat_conjugate(global_base_rot), global_body_rot[body_name])
                else:
                    # upper body
                    pass
            return global_joint_pos, global_joint_rot, local_joint_pos, local_joint_rot



    
class AMPLoaderMorph:
    FRAME_TIME = 0
    MODEL_DOF = 33
    REFERENCE_START_INDEX = None
    REFERENCE_END_INDEX = None


    POS_SIZE = 3
    JOINT_POS_SIZE = 12
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    JOINT_VEL_SIZE = 12
    ROOT_POS_SIZE = 1 # base height 
    ROOT_ROT_SIZE = 4 # projected gravity

    OBSERVATION_DIM = ROOT_POS_SIZE + ROOT_ROT_SIZE + LINEAR_VEL_SIZE + ANGULAR_VEL_SIZE\
         + JOINT_POS_SIZE + JOINT_VEL_SIZE + 2*(POS_SIZE)

    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + ROOT_POS_SIZE

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROOT_ROT_SIZE

    LINEAR_VEL_START_IDX = ROOT_ROT_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE

    JOINT_POSE_START_IDX = ANGULAR_VEL_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

    FOOT_POS_START_IDX = JOINT_VEL_END_IDX
    FOOT_POS_END_IDX = FOOT_POS_START_IDX + 2*POS_SIZE


    HZ = None
    # # REFERENCE_START_INDEX = [int(5.6*HZ), int(4.4*HZ)]
    # # REFERENCE_END_INDEX = [int(7.4005*HZ), int(5.6005*HZ)]
    # # REFERENCE_START_INDEX = int(2.*HZ)
    # # REFERENCE_END_INDEX = int(5.6005*HZ)
    # REFERENCE_START_INDEX = int(2.0*HZ)
    # REFERENCE_END_INDEX = int(5.6005*HZ)


    #CMU
    # HZ = 120
    # REFERENCE_START_INDEX = int(0.*HZ)
    # REFERENCE_END_INDEX = int(3.0*HZ)



    def __init__(
            self,
            device,
            time_between_frames,
            preload_transitions=False,
            num_preload_transitions=1000000,
            reference_dict={}, play=False, iterations=3000,
            num_morphology=1,
            ):
        """Expert dataset provides AMP observations from Dog mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.play = play
        self.iterations = iterations
        self.time_between_frames = time_between_frames
        self.num_morphology = num_morphology
        # Values to store for each trajectory.
        self.reference_dict = reference_dict
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []
        self.morph_params = [] # len will be num_morphologies
        self.num_preload_transitions = num_preload_transitions
        for i, (jsonf, info) in enumerate(reference_dict.items()):
            motion_file = jsonf
            model_file = info['xml']
            hz = info['hz']
            reference_start_index = int(hz * info['start_time'])
            reference_end_index = int(hz * info['end_time'])
            AMPLoaderMorph.MODEL_DOF = info['model_dof']
            AMPLoaderMorph.REFERENCE_START_INDEX = int(reference_start_index)
            AMPLoaderMorph.REFERENCE_END_INDEX = int(reference_end_index)
            AMPLoaderMorph.HZ = hz
            motion_weight = info['weight']
            
            print(motion_file)
            self.trajectory_names.append(motion_file.split('.')[0])
            with open(motion_file, "r") as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"]) # motion data in json form.

                self.trajectory_idxs.append(i)
                self.trajectory_weights.append(
                    # float(motion_json["MotionWeight"]))
                    float(motion_weight))
                frame_duration = float(motion_data[1,AMPLoaderMorph.FRAME_TIME] - motion_data[0,AMPLoaderMorph.FRAME_TIME])
                self.trajectory_frame_durations.append(frame_duration)
                traj_len = (reference_end_index-reference_start_index - 1) * frame_duration
                # traj_len = (motion_data.shape[0] - 1) * frame_duration
                self.trajectory_lens.append(traj_len)
                # self.trajectory_num_frames.append(float(motion_data.shape[0]))
                self.trajectory_num_frames.append(float(reference_end_index - reference_start_index))

            print(f"Loaded {traj_len}s. motion from {motion_file}.")
            print(f"Size of Reference Observation : {self.observation_dim}")
        
        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        self.preload_transitions = preload_transitions



    def register_morphology(self, target_model_file, morph_params):
        for i, (jsonf, info) in enumerate(self.reference_dict.items()):
            motion_file = jsonf
            model_file = info['xml']
            hz = info['hz']
            reference_start_index = int(hz * info['start_time'])
            reference_end_index = int(hz * info['end_time'])
            AMPLoaderMorph.MODEL_DOF = info['model_dof']
            AMPLoaderMorph.REFERENCE_START_INDEX = int(reference_start_index)
            AMPLoaderMorph.REFERENCE_END_INDEX = int(reference_end_index)
            AMPLoaderMorph.HZ = hz
            motion_weight = info['weight']

            with open(motion_file, "r") as f:
                motion_json = json.load(f)
                motion_data = np.array(motion_json["Frames"]) # motion data in json form.
                if (model_file == target_model_file or model_file == ''): 
                    print("No retargetting motion")
                    processed_data_full, info = self.process_data(motion_data, i)
                    processed_data_joints = processed_data_full[:, self.JOINT_POSE_START_IDX:self.JOINT_VEL_END_IDX]

                else:
                    print("RETARGET REFERENCE MOTIONS")
                    print("Target file : ", target_model_file)
                    print("Source file : ", info['xml'])
                    motion_retarget = self.MotionRetarget(source_model_path=model_file, target_model_path=target_model_file, device=self.device) # model_file is the source
                    processed_data_full, _ = motion_retarget.retarget(motion_data, i, self.play, self.iterations)
                    processed_data_full = processed_data_full.to(torch.float16)
                    processed_data_joints = processed_data_full[:, self.JOINT_POSE_START_IDX:self.JOINT_VEL_END_IDX]

                if len(self.trajectories) < len(self.reference_dict):
                    self.trajectories.append(
                        processed_data_joints.unsqueeze(1)
                    )
                    self.trajectories_full.append(
                        processed_data_full.unsqueeze(1)
                    )
                else:
                    self.trajectories[i] = torch.cat((self.trajectories[i], processed_data_joints.unsqueeze(1)), dim=1)
                    self.trajectories_full[i] = torch.cat((self.trajectories_full[i], processed_data_full.unsqueeze(1)), dim=1)
        self.morph_params.append(torch.tensor(morph_params, device=self.device, dtype=torch.float16))

    def post_registering_morphologies(self):
        if self.preload_transitions:
            print(f'Preloading {self.num_preload_transitions} transitions')
            traj_idxs = self.weighted_traj_idx_sample_batch(self.num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times) 
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print(f'Finished preloading')

        self.all_trajectories_full = torch.vstack(self.trajectories_full)
        torch.save({'trajectory': self.trajectories[0], 'morphology': self.morph_params}, '/home/cha/isaac_ws/AMP_for_hardware/legged_gym/scripts/trajectory.pt')

    def morphology_idx_sample(self):
        return np.random.choice([i for i in range(self.num_morphology)])
            
    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(
            self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights,
            replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(
            0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, dtype=torch.float16, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, dtype=torch.float16, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32, requires_grad=False).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        all_frame_pos_starts = torch.zeros(len(traj_idxs), self.num_morphology,AMPLoaderMorph.ROOT_POS_SIZE, dtype=torch.float16, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), self.num_morphology,AMPLoaderMorph.ROOT_POS_SIZE, dtype=torch.float16, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), self.num_morphology, AMPLoaderMorph.ROOT_ROT_SIZE, dtype=torch.float16, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), self.num_morphology, AMPLoaderMorph.ROOT_ROT_SIZE, dtype=torch.float16, device=self.device)
        all_frame_linvel_starts = torch.zeros(len(traj_idxs), self.num_morphology,AMPLoaderMorph.LINEAR_VEL_SIZE, dtype=torch.float16, device=self.device)
        all_frame_linvel_ends = torch.zeros(len(traj_idxs), self.num_morphology,AMPLoaderMorph.LINEAR_VEL_SIZE, dtype=torch.float16, device=self.device)
        all_frame_angvel_starts = torch.zeros(len(traj_idxs), self.num_morphology,AMPLoaderMorph.ANGULAR_VEL_SIZE, dtype=torch.float16, device=self.device)
        all_frame_angvel_ends = torch.zeros(len(traj_idxs), self.num_morphology,AMPLoaderMorph.ANGULAR_VEL_SIZE, dtype=torch.float16, device=self.device)
        all_frame_amp_starts = torch.zeros(len(traj_idxs), self.num_morphology, AMPLoaderMorph.JOINT_VEL_END_IDX - AMPLoaderMorph.JOINT_POSE_START_IDX, dtype=torch.float16, device=self.device)
        all_frame_amp_ends = torch.zeros(len(traj_idxs), self.num_morphology,  AMPLoaderMorph.JOINT_VEL_END_IDX - AMPLoaderMorph.JOINT_POSE_START_IDX, dtype=torch.float16, device=self.device)
        all_frame_foot_pos_starts = torch.zeros(len(traj_idxs), self.num_morphology, AMPLoaderMorph.FOOT_POS_END_IDX - AMPLoaderMorph.FOOT_POS_START_IDX, dtype=torch.float16, device=self.device)
        all_frame_foot_pos_ends = torch.zeros(len(traj_idxs), self.num_morphology, AMPLoaderMorph.FOOT_POS_END_IDX - AMPLoaderMorph.FOOT_POS_START_IDX, dtype=torch.float16, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask, :] = AMPLoaderMorph.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask, :] = AMPLoaderMorph.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask, :] = AMPLoaderMorph.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask, :] = AMPLoaderMorph.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_linvel_starts[traj_mask, :] = AMPLoaderMorph.get_linear_vel_batch(trajectory[idx_high[traj_mask]])
            all_frame_linvel_ends[traj_mask, :] = AMPLoaderMorph.get_linear_vel_batch(trajectory[idx_low[traj_mask]])
            all_frame_angvel_starts[traj_mask, :] = AMPLoaderMorph.get_angular_vel_batch(trajectory[idx_high[traj_mask]])
            all_frame_angvel_ends[traj_mask, :] = AMPLoaderMorph.get_angular_vel_batch(trajectory[idx_low[traj_mask]])
            all_frame_amp_starts[traj_mask, :] = trajectory[idx_low[traj_mask]][:, :, AMPLoaderMorph.JOINT_POSE_START_IDX:AMPLoaderMorph.JOINT_VEL_END_IDX]
            all_frame_amp_ends[traj_mask, :] = trajectory[idx_high[traj_mask]][:, :, AMPLoaderMorph.JOINT_POSE_START_IDX:AMPLoaderMorph.JOINT_VEL_END_IDX]
            all_frame_foot_pos_starts[traj_mask, :] = AMPLoaderMorph.get_foot_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_foot_pos_ends[traj_mask, :] = AMPLoaderMorph.get_foot_pos_batch(trajectory[idx_high[traj_mask]])

        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float16, requires_grad=False).unsqueeze(-1).unsqueeze(-1)

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend) 
        rot_blend = self.slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        lin_blend = self.slerp(all_frame_linvel_starts, all_frame_linvel_ends, blend)
        ang_blend = self.slerp(all_frame_angvel_starts, all_frame_angvel_ends, blend)
        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        foot_pos_blend = self.slerp(all_frame_foot_pos_starts, all_frame_foot_pos_ends, blend)

        ret = torch.cat([pos_blend, rot_blend, lin_blend, ang_blend, amp_blend, foot_pos_blend], dim=-1) 
        # ret = torch.cat([pos_blend, rot_blend, amp_blend, foot_pos_blend], dim=-1) 
        # ret = torch.cat([rot_blend, amp_blend, foot_pos_blend], dim=-1) 
        return ret

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(
                self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        root_pos0, root_pos1 = AMPLoaderMorph.get_root_pos(frame0), AMPLoaderMorph.get_root_pos(frame1)
        root_rot0, root_rot1 = AMPLoaderMorph.get_root_rot(frame0), AMPLoaderMorph.get_root_rot(frame1)
        linear_vel_0, linear_vel_1 = AMPLoaderMorph.get_linear_vel(frame0), AMPLoaderMorph.get_linear_vel(frame1)
        angular_vel_0, angular_vel_1 = AMPLoaderMorph.get_angular_vel(frame0), AMPLoaderMorph.get_angular_vel(frame1)
        joints0, joints1 = AMPLoaderMorph.get_joint_pose(frame0), AMPLoaderMorph.get_joint_pose(frame1)
        joint_vel_0, joint_vel_1 = AMPLoaderMorph.get_joint_vel(frame0), AMPLoaderMorph.get_joint_vel(frame1)
        foot_pos0, foot_pos1 = AMPLoaderMorph.get_foot_pos(frame0), AMPLoaderMorph.get_foot_pos(frame1)
        
        blend_root_pos = self.slerp(root_pos0, root_pos1, blend) 
        blend_root_rot = self.slerp(root_rot0, root_rot1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)
        blend_foot_pos = self.slerp(foot_pos0, foot_pos1, blend)

        ret = torch.cat([
            blend_root_pos, blend_root_rot, blend_linear_vel, blend_angular_vel, blend_joints, blend_joints_vel,
            blend_foot_pos
            ], dim=-1)
        # ret = torch.cat([ 
        #     blend_root_pos, blend_root_rot, blend_joints, blend_joints_vel,
        #     blend_foot_pos
        #     ], dim=-1)
        # ret = torch.cat([
        #     blend_root_rot, blend_joints, blend_joints_vel,
        #     blend_foot_pos
        #     ], dim=-1)

        return ret

    def feed_forward_generator(self, num_mini_batch, mini_batch_size, morphnet = False):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(
                    self.preloaded_s.shape[0], size=mini_batch_size)
                s_morphs = self.preloaded_s[idxs]
                s_next_morphs = self.preloaded_s_next[idxs]
                morph_idxs = np.random.choice(
                    self.num_morphology, size=mini_batch_size
                )
                morph_params_tensor = torch.stack(self.morph_params)
                # Select the appropriate indices
                morph_params = morph_params_tensor[morph_idxs]
                s = s_morphs[torch.arange(mini_batch_size), morph_idxs, :]
                s_next = s_next_morphs[torch.arange(mini_batch_size), morph_idxs, :]
                # s = torch.cat((s, morph_params), dim=-1)
                if morphnet:
                    s_next = torch.cat((s_next, morph_params), dim=-1)
            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    morph_idx = np.random.choice(self.num_morphology)
                    morph_params_tensor = torch.stack(self.morph_params)
                    # Select the appropriate indices
                    morph_params = morph_params_tensor[morph_idxs]
                    s.append(torch.cat((self.get_full_frame_at_time(traj_idx, frame_time)[morph_idx], morph_params), dim=-1))
                    if morphnet:
                        s_next.append(
                            torch.cat(
                            (self.get_full_frame_at_time(
                                traj_idx, frame_time + self.time_between_frames)[morph_idx], morph_params), dim=-1)
                            )
                    else:
                        s_next.append(
                            self.get_full_frame_at_time(
                                traj_idx, frame_time + self.time_between_frames)[morph_idx]
                        )
                s = torch.vstack(s)
                s_next = torch.vstack(s_next)
            yield s.to(torch.float32), s_next.to(torch.float32)

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.OBSERVATION_DIM

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_root_pos(pose):
        return pose[:,AMPLoaderMorph.ROOT_POS_START_IDX:AMPLoaderMorph.ROOT_POS_END_IDX]

    def get_root_pos_batch(poses):
        return poses[:,:, AMPLoaderMorph.ROOT_POS_START_IDX:AMPLoaderMorph.ROOT_POS_END_IDX]

    def get_root_rot(pose):
        return pose[:, AMPLoaderMorph.ROOT_ROT_START_IDX:AMPLoaderMorph.ROOT_ROT_END_IDX]

    def get_root_rot_batch(poses):
        return poses[:, :, AMPLoaderMorph.ROOT_ROT_START_IDX:AMPLoaderMorph.ROOT_ROT_END_IDX]

    def get_linear_vel(pose):
        return pose[:,AMPLoaderMorph.LINEAR_VEL_START_IDX:AMPLoaderMorph.LINEAR_VEL_END_IDX]
    
    def get_linear_vel_batch(poses):
        return poses[:,:, AMPLoaderMorph.LINEAR_VEL_START_IDX:AMPLoaderMorph.LINEAR_VEL_END_IDX]

    def get_angular_vel(pose):
        return pose[:,AMPLoaderMorph.ANGULAR_VEL_START_IDX:AMPLoaderMorph.ANGULAR_VEL_END_IDX]  

    def get_angular_vel_batch(poses):
        return poses[:,:, AMPLoaderMorph.ANGULAR_VEL_START_IDX:AMPLoaderMorph.ANGULAR_VEL_END_IDX]
    
    def get_joint_pose(pose):
        return pose[:, AMPLoaderMorph.JOINT_POSE_START_IDX:AMPLoaderMorph.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(poses):
        return poses[:, :, AMPLoaderMorph.JOINT_POSE_START_IDX:AMPLoaderMorph.JOINT_POSE_END_IDX]
  
    def get_joint_vel(pose):
        return pose[:, AMPLoaderMorph.JOINT_VEL_START_IDX:AMPLoaderMorph.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(poses):
        return poses[:, :, AMPLoaderMorph.JOINT_VEL_START_IDX:AMPLoaderMorph.JOINT_VEL_END_IDX]
          
    def get_foot_pos(pose):
        return pose[:, AMPLoaderMorph.FOOT_POS_START_IDX:AMPLoaderMorph.FOOT_POS_END_IDX]

    def get_foot_pos_batch(poses):
        return poses[:, :, AMPLoaderMorph.FOOT_POS_START_IDX:AMPLoaderMorph.FOOT_POS_END_IDX]

    @staticmethod
    def process_data(traj: np.ndarray, index: int):# Process raw data from Mujoco
        traj = denoise(traj, window_length=int(0.2*AMPLoaderMorph.HZ), polyorder=2)
        time = traj[AMPLoaderMorph.REFERENCE_START_INDEX:AMPLoaderMorph.REFERENCE_END_INDEX,0]
        pelv_pos = traj[AMPLoaderMorph.REFERENCE_START_INDEX:AMPLoaderMorph.REFERENCE_END_INDEX, 1:4]
        pelv_rpy = traj[AMPLoaderMorph.REFERENCE_START_INDEX:AMPLoaderMorph.REFERENCE_END_INDEX, 4:7] # roll, pitch, yaw order
        pelv_vel = traj[AMPLoaderMorph.REFERENCE_START_INDEX:AMPLoaderMorph.REFERENCE_END_INDEX, 7:13]
        q_pos = traj[AMPLoaderMorph.REFERENCE_START_INDEX:AMPLoaderMorph.REFERENCE_END_INDEX, 13:13+AMPLoaderMorph.JOINT_POS_SIZE]
        start = 13+AMPLoaderMorph.MODEL_DOF
        end = start + AMPLoaderMorph.JOINT_VEL_SIZE
        q_vel = traj[AMPLoaderMorph.REFERENCE_START_INDEX:AMPLoaderMorph.REFERENCE_END_INDEX, start:end]
        start += AMPLoaderMorph.MODEL_DOF
        end = start + 3
        L_foot_pos = traj[AMPLoaderMorph.REFERENCE_START_INDEX:AMPLoaderMorph.REFERENCE_END_INDEX, start:end]
        start = end
        end = start + 3
        L_foot_rpy = traj[AMPLoaderMorph.REFERENCE_START_INDEX:AMPLoaderMorph.REFERENCE_END_INDEX, start:end]
        start = end
        end = start + 3
        R_foot_pos = traj[AMPLoaderMorph.REFERENCE_START_INDEX:AMPLoaderMorph.REFERENCE_END_INDEX, start:end]
        start=end
        end = start + 3
        R_foot_rpy = traj[AMPLoaderMorph.REFERENCE_START_INDEX:AMPLoaderMorph.REFERENCE_END_INDEX, start:end]
        assert end == 1+3+3+6+AMPLoaderMorph.MODEL_DOF+AMPLoaderMorph.MODEL_DOF+3+3+3+3, f"process data, shape mismatch {end}"

        # torch versions of data
        pelv_pos_torch = to_torch(pelv_pos, dtype=torch.float32)
        pelv_vel_torch = to_torch(pelv_vel, dtype=torch.float32)
        q_pos_torch = to_torch(q_pos, dtype=torch.float32)
        q_vel_torch = to_torch(q_vel, dtype=torch.float32)
        L_foot_pos_torch = to_torch(L_foot_pos, dtype=torch.float32)
        L_foot_rpy_torch = to_torch(L_foot_rpy, dtype=torch.float32)
        R_foot_pos_torch = to_torch(R_foot_pos, dtype=torch.float32)
        R_foot_rpy_torch = to_torch(R_foot_rpy, dtype=torch.float32)

        # Process raw data
        pelvis_yaw =pelv_rpy[:, 2]
        pelvis_pitch = pelv_rpy[:, 1]
        pelvis_roll = pelv_rpy[:, 0]
        pelvis_quat = quat_from_euler_xyz(to_torch(pelvis_roll, dtype=torch.float32), to_torch(pelvis_pitch, dtype=torch.float32), to_torch(pelvis_yaw, dtype=torch.float32)) # tensor
        base_pos_global_torch = pelv_pos_torch

        # Create AMP observation
        base_height = base_pos_global_torch[:, 2]
        assert (~torch.isfinite(base_height)).sum() == 0, "Found non finite element"
        assert (~torch.isfinite(pelvis_quat)).sum() == 0, "Found non finite element 0"
        # gravity_vector = to_torch(get_axis_params(-1., 2), device=self.device).repeat((pelvis_yaw.shape[0], 1))
        # projected_gravity = quat_rotate_inverse(pelvis_quat, gravity_vector)
        base_lin_vel = quat_rotate_inverse(pelvis_quat, pelv_vel_torch[:, :3]) 
        assert (~torch.isfinite(base_lin_vel)).sum() == 0, "Found non finite element 1"
        base_ang_vel = quat_rotate_inverse(pelvis_quat, pelv_vel_torch[:, 3:])
        assert (~torch.isfinite(base_ang_vel)).sum() == 0, "Found non finite element 2"
        L_foot_pos_base_torch = quat_rotate_inverse(pelvis_quat, L_foot_pos_torch - base_pos_global_torch)
        assert (~torch.isfinite(L_foot_pos_base_torch)).sum() == 0, "Found non finite element 3"
        R_foot_pos_base_torch = quat_rotate_inverse(pelvis_quat, R_foot_pos_torch - base_pos_global_torch) 
        assert (~torch.isfinite(R_foot_pos_base_torch)).sum() == 0, "Found non finite element 4"
        L_foot_quat_global_torch = quat_from_euler_xyz(L_foot_rpy_torch[:, 0], L_foot_rpy_torch[:, 1], L_foot_rpy_torch[:, 2])
        assert (~torch.isfinite(L_foot_quat_global_torch)).sum() == 0, "Found non finite element 5"
        R_foot_quat_global_torch = quat_from_euler_xyz(R_foot_rpy_torch[:, 0], R_foot_rpy_torch[:, 1], R_foot_rpy_torch[:, 2])
        assert (~torch.isfinite(R_foot_quat_global_torch)).sum() == 0, "Found non finite element 6"
        L_foot_quat_base_torch = quat_mul(quat_conjugate(pelvis_quat), L_foot_quat_global_torch)
        assert (~torch.isfinite(L_foot_quat_base_torch)).sum() == 0, "Found non finite element 7"
        R_foot_quat_base_torch = quat_mul(quat_conjugate(pelvis_quat), R_foot_quat_global_torch)
        assert (~torch.isfinite(R_foot_quat_base_torch)).sum() == 0, "Found non finite element 8"
        ret = torch.concat((base_height.unsqueeze(-1), pelvis_quat, base_lin_vel, base_ang_vel, q_pos_torch, q_vel_torch, L_foot_pos_base_torch, R_foot_pos_base_torch), dim=-1) 
        # ret = torch.concat((base_height.unsqueeze(-1), pelvis_quat, q_pos_torch, q_vel_torch, L_foot_pos_base_torch, R_foot_pos_base_torch), dim=-1) 
        # ret = torch.concat((pelvis_quat, q_pos_torch, q_vel_torch, L_foot_pos_base_torch, R_foot_pos_base_torch), dim=-1) 
        info = torch.concat((base_height.unsqueeze(-1), base_lin_vel, base_ang_vel, L_foot_quat_base_torch, R_foot_quat_base_torch), dim=-1)
        with open(os.path.join(PROCESSED_DATA_DIR, 'processed_data_'+str(index)+'.txt'), "w") as file:
            for line in ret.cpu().numpy().tolist():
                file.write(' '.join(map(str, line)) + '\n')
            print(f"Processed data {index} written to txt file at "+PROCESSED_DATA_DIR)
        return ret, info
    
    class MotionRetarget():

        JOINT_MAPPING = {
            'virtual_joint': 0,
            'L_HipYaw_Joint': 7, 
            'L_HipRoll_Joint': 8,
            'L_HipPitch_Joint': 9,
            'L_Knee_Joint': 10,
            'L_AnklePitch_Joint': 11,
            'L_AnkleRoll_Joint': 12,

            'R_HipYaw_Joint': 13, 
            'R_HipRoll_Joint': 14,
            'R_HipPitch_Joint': 15,
            'R_Knee_Joint': 16,
            'R_AnklePitch_Joint': 17,
            'R_AnkleRoll_Joint': 18,
        }

        def __init__(self, source_model_path: str, target_model_path: str, device):
            source = source_model_path
            target = target_model_path
            source_bodies, source_joints, source_edges = self.process_model(source)
            target_bodies, target_joints, target_edges = self.process_model(target)
            self.root_ratio = target_bodies['base_link']['info']['position'][-1] / source_bodies['base_link']['info']['position'][-1]
            assert self.root_ratio < 10 and self.root_ratio > 0.1, f"Root ratio is abnormally small, {self.root_ratio}"
            self.source = {
                'bodies': source_bodies,
                'joints': source_joints,
                'edges': source_edges
            }
            self.target = {
                'bodies': target_bodies,
                'joints': target_joints,
                'edges': target_edges
            }
            self.source_model_path = source_model_path
            self.target_model_path = target_model_path
            # print("SOURCE MODEL")
            # self.print_model(self.source)
            # print("TARGET MODEL")
            # self.print_model(self.target)
            self.device = device
            

        def process_model(self, model_path: str):
            tree = etree.parse(model_path)
            root = tree.getroot()
            joints = {}
            bodies = {}
            edges = {}

        # Collect body and joint data
            for body in root.findall('.//body'):
                body_name = body.get('name')
                pos = np.array([float(num) for num in body.get('pos', '0 0 0').split()], dtype=float) # default value is 0, 0, 0
                quat = np.array([float(num) for num in body.get('quat', '1 0 0 0').split()], dtype=float) # default value is 0, 0, 0
                if body.find('inertial') is not None: # not a dummy
                    inertial = body.find('inertial')
                    mass = np.array([float(num) for num in inertial.get('mass', '0.').split()], dtype=float) # default value is 0, 0, 0
                    com = np.array([float(num) for num in inertial.get('pos', '0 0 0').split()], dtype=float) # default value is 0, 0, 0
                    inertia = np.array([float(num) for num in inertial.get('fullinertia', '0 0 0 0 0 0').split()], dtype=float) # default value is 0, 0, 0
                else:
                    mass = None
                    com = None
                    inertia = None

                body_info = {
                    'position': pos,
                    'quat': quat,
                    'mass': mass,
                    'inertia': inertia,
                    'com': com
                }

                parent = body.getparent()
                parent_name = parent.get('name')
                if parent_name is None:
                    parent_name = 'world'
                
                if body.find('joint') is not None:
                    for i, joint in enumerate(body.findall('joint')):
                        joint_name = joint.get('name')
                        joint_pos = np.array([float(num) for num in joint.get('pos', '0 0 0').split()], dtype=float)
                        joint_axis = np.array([float(num) for num in joint.get('axis', '0 0 1').split()], dtype=float)
                        joint_quat = np.array([float(num) for num in joint.get('quat', '1 0 0 0').split()], dtype=float)
                        joint_type = joint.get('type', 'hinge')
                        joint_range = np.array([float(num) for num in joint.get('range', '-3.14 3.14').split()], dtype=float)
                        joint_info = {
                            'parent' : parent_name,
                            'child' : body_name,
                            'position' : joint_pos,
                            'axis' : joint_axis,
                            'type' : joint_type,
                            'range' : joint_range,
                            'quat' : joint_quat
                        }
                        joints[joint_name] = joint_info
                        # process link length between joint_name and parent joint
                        edges[joint_name] = np.linalg.norm(body_info['position'], ord=2, axis=-1)
                else: # fixed link
                    joint_name = None             
                bodies[body_name] = {'parent': parent_name, 'joint': joint_name, 'info' : body_info}    
                m = {'bodies': bodies, 'joints': joints}

            return bodies, joints, edges

        def print_model(self, model):
            bodies, joints, edges = model['bodies'], model['joints'], model['edges']
            print("PRINTING MODEL")
            print("---------------INFO----------------")
            for body_name, val in bodies.items():
                print("Link name : ", body_name)
                print(val)
                print("Joint name : ", val['joint'])
                print(joints[val['joint']]) if val['joint'] is not None else print("None")
                print("----------------------")
            print("-------------Link length-----------")
            for joint_name, val in edges.items():
                if joints[joint_name]['child'] != "base_link":
                    print(f"{joint_name} and {bodies[joints[joint_name]['parent']]['joint']} : {val}")

        def retarget_tensor(self, reference: torch.Tensor, iterations :int,  play=False) -> torch.Tensor: # outputs retargetted reference
            dtype=reference.dtype
            length_ratios = {}
            reference_length = reference.shape[0]
            observation_dim = reference.shape[-1]
            for key, val in self.target['edges'].items():
                if key in self.JOINT_MAPPING:
                    if self.source['edges'][key] == 0:
                        length_ratios[key] = 0.
                    else:
                        length_ratios[key] = val/self.source['edges'][key]
            source_global_joint_pos = {}
            source_global_joint_quat = {}
            target_global_joint_pos = {}
            target_local_joint_pos = {}
            target_global_joint_quat = {}
            target_local_joint_quat = {}
            relative_position_vectors = {}
            source_bodies, source_joints= self.source['bodies'], self.source['joints']
            target_bodies, target_joints = self.target['bodies'], self.target['joints']

            start = 0
            end = start+AMPLoaderMorph.ROOT_POS_SIZE
            # source_global_root_pos = torch.cat((torch.zeros_like(reference[:, start:end]), torch.zeros_like(reference[:, start:end]), reference[:, start:end]), dim=-1)
            source_global_root_pos = torch.cat((torch.zeros_like(reference[:, 0:1], dtype=dtype), torch.zeros_like(reference[:, 0:1], dtype=dtype), torch.zeros_like(reference[:, 0:1], dtype=dtype)), dim=-1).to(self.device)
            start = end
            end = start+AMPLoaderMorph.ROOT_ROT_SIZE
            source_global_root_quat = reference[:, start:end]
            start = end
            end = start+AMPLoaderMorph.LINEAR_VEL_SIZE
            source_local_base_lin_vel = reference[:, start:end]
            start = end
            end = start+AMPLoaderMorph.ANGULAR_VEL_SIZE
            source_local_base_ang_vel = reference[:, start:end]
            start = end
            end = start+AMPLoaderMorph.JOINT_POS_SIZE
            source_qpos = reference[:, start:end]
            start = end
            end = start+AMPLoaderMorph.JOINT_VEL_SIZE
            source_qvel = reference[:, start:end]
            start = end
            end = start+AMPLoaderMorph.POS_SIZE
            source_local_Lfoot_pos = reference[:, start:end]
            start = end
            end = start+AMPLoaderMorph.POS_SIZE
            source_local_Rfoot_pos = reference[:, start:end]
            assert end == reference.shape[-1], f"Retarget, observation shape do not match. Must be {reference.shape[-1]} but is {end}"
            
            source_q = torch.concat((source_global_root_pos, source_global_root_quat, reference[:, AMPLoaderMorph.JOINT_POSE_START_IDX:AMPLoaderMorph.JOINT_POSE_END_IDX]), dim=-1)
            assert source_q.shape == (reference_length, 7+12)
            source_global_joint_pos, source_global_joint_quat, source_local_joint_pos, source_local_joint_quat = self.forward_kinematics(self.source, source_q, self.JOINT_MAPPING)
            for body_name, body in source_bodies.items():

                parent_name = body['parent']
                joint_name = body['joint']
                body_info = body['info']

                if joint_name in self.JOINT_MAPPING and body_name != "base_link":
                    if 'HipYaw_Joint' in joint_name:
                        relative_position_vectors[joint_name] = to_torch(length_ratios[joint_name], device=self.device,dtype=dtype).tile((reference_length, 1)) * \
                            (source_global_joint_pos[joint_name] - source_global_joint_pos['virtual_joint']) # scaled relative position vector from parent joint to current joint in global frame                    
                    else:
                        relative_position_vectors[joint_name] = to_torch(length_ratios[joint_name], device=self.device,dtype=dtype).tile((reference_length, 1)) * \
                            (source_global_joint_pos[joint_name] - source_global_joint_pos[source_bodies[parent_name]['joint']]) # scaled relative position vector from parent joint to current joint in global frame

            # 3. Starting from base link, sum up the relative position vectors and transform back to base frame coordinates              
            joint_pos_global_retarget = torch.zeros((reference_length, 3 * 13), dtype=dtype, device=self.device)
            joint_pos_local_retarget = torch.zeros((reference_length, 3 * 12), dtype=dtype, device=self.device)
            joint_quat_global_retarget = torch.zeros((reference_length, 4*13), dtype=dtype, device=self.device)
            joint_quat_local_retarget = torch.zeros((reference_length, 4*12), dtype=dtype, device=self.device)
            for body_name, body in source_bodies.items():
                parent_name = body['parent']
                joint_name = body['joint']
                body_info = body['info']
                base_pos = source_global_joint_pos[source_bodies['base_link']['joint']]
                base_quat = source_global_joint_quat[source_bodies['base_link']['joint']]

                if body_name == "base_link":
                    target_global_joint_pos[joint_name] = base_pos
                    target_global_joint_quat[joint_name] = base_quat
                    joint_pos_global_retarget[:, 0:3] = base_pos
                elif joint_name in self.JOINT_MAPPING and body_name != "base_link":
                    parent_joint_name = source_bodies[parent_name]['joint']
                    if 'HipYaw_Joint' in joint_name:
                        parent_joint_name = 'virtual_joint'
                    target_global_joint_pos[joint_name] = target_global_joint_pos[parent_joint_name] + relative_position_vectors[joint_name]
                    target_global_joint_quat[joint_name] = source_global_joint_quat[joint_name]
                    target_local_joint_pos[joint_name] = quat_rotate_inverse(base_quat, (target_global_joint_pos[joint_name] - base_pos)) # relative position of joint from base in base frame coordinates
                    target_local_joint_quat[joint_name] = quat_mul(quat_conjugate(base_quat), target_global_joint_quat[joint_name]) # quat of joint frame in base frame coordinates. Only the foot orientation will be a constraint for the IK
                    joint_pos_global_retarget[:, 3*(self.JOINT_MAPPING[joint_name]-6):3*(self.JOINT_MAPPING[joint_name]-5)] = target_global_joint_pos[joint_name]
                    joint_pos_local_retarget[:, 3*(self.JOINT_MAPPING[joint_name]-7):3*(self.JOINT_MAPPING[joint_name]-6)] = target_local_joint_pos[joint_name]
                    joint_quat_global_retarget[:, 4*(self.JOINT_MAPPING[joint_name]-6):4*(self.JOINT_MAPPING[joint_name]-5)] = target_global_joint_quat[joint_name]
                    joint_quat_local_retarget[:, 4*(self.JOINT_MAPPING[joint_name]-7):4*(self.JOINT_MAPPING[joint_name]-6)] = target_local_joint_quat[joint_name]

            # # 4. Solve numerical IK from joint positions, foot positions and orientation
            weight_joint_pos, weight_joint_pose, weight_joint_vel = 1., .5, .1
            # weight_joint_pos, weight_foot_quat, weight_joint_pose, weight_hip_pose, weight_joint_vel = 1.2, 0.1, .1, 0.1, .1 For CMU refernece
            weight_norm = (weight_joint_pos+weight_joint_pose)
            weight_joint_pos /= weight_norm
            weight_joint_pose/= weight_norm
            weight_joint_vel /= weight_norm

            def cost_function(x):
                # x = [q for each joints] -> shape of 12
                # Initialize x as [0, reference[:, ROOT_ROT_START_IDX:ROOT_ROT_END_IDX], reference[:, JOINT_POSE_START_IDX:JOINT_POSE_END_IDX]]
                # joint_pos_global_retarget = [base_pos, joint_pos] in global frame coordinates
                # joint_pos_local_retarget = [joint_pos] in base local frame coordinates

                temp = torch.zeros((x.shape[0], 7), requires_grad=False, device=self.device, dtype=torch.float32)
                temp[:, -1] = 1.
                _, x_global_joint_quat, x_local_joint_pos, x_local_joint_quat = self.forward_kinematics(self.target, torch.cat((temp, x), dim=-1), self.JOINT_MAPPING)
                x_local_joint_pos_tensor = torch.zeros_like(joint_pos_local_retarget, dtype=torch.float32)
                x_local_joint_quat_tensor = torch.zeros_like(joint_quat_local_retarget)
                for joint_name, joint_index in self.JOINT_MAPPING.items():
                    if key == 'virtual_joint':
                        pass
                    else:
                        index = joint_index-7
                        x_local_joint_pos_tensor[:, 3*index:3*(index+1)] = x_local_joint_pos[joint_name]
                        x_local_joint_quat_tensor[:, 4*index:4*(index+1)] = x_local_joint_quat[joint_name]
                assert x_local_joint_pos_tensor.shape == (reference_length, 36), f"local joint pos tensor shape is {x_local_joint_pos_tensor.shape}"

                # mask_ref = [i+JOINT_POSE_START_IDX for i in mask]
                # joint_pose_cost = torch.nn.MSELoss()(x[:, mask], TOCABI_INIT_POS_TORCH[:, mask]) # For CMU reference
                joint_pose_cost = torch.nn.MSELoss()(x, source_qpos)
                joint_pos_cost = torch.nn.MSELoss()(x_local_joint_pos_tensor, joint_pos_local_retarget)
                joint_vel_cost = torch.nn.MSELoss()(x[1:], x[:-1])
    # 
                L_ankle_roll_joint = FOOT_NAME[0]
                R_ankle_roll_joint = FOOT_NAME[1]

                x_Lfoot_quat = x_global_joint_quat[L_ankle_roll_joint]
                x_Rfoot_quat = x_global_joint_quat[R_ankle_roll_joint]
                zero_quat = torch.tensor([0, 0, 0, 1], dtype=dtype, device=self.device, requires_grad=False).tile(reference_length,1)
                # foot_quat_cost = (distance_between_quats(zero_quat, x_Lfoot_quat, neglect_axis=2).pow(2) \
                #                   + distance_between_quats(zero_quat, x_Rfoot_quat, neglect_axis=2).pow(2)).mean()
                cosine_loss = torch.nn.CosineSimilarity(dim=-1)

                total_cost = weight_joint_pos*joint_pos_cost + weight_joint_pose*joint_pose_cost + weight_joint_vel*joint_vel_cost
                return torch.sum(total_cost), joint_pos_cost, joint_pose_cost, (1 - cosine_loss(zero_quat, x_Lfoot_quat)).mean(), (1 - cosine_loss(zero_quat, x_Rfoot_quat)).mean()
            
            n_iterations = iterations
            if play:
                n_iterations = 0
            # q_opt = torch.zeros((reference_length, 12), dtype=torch.float32, requires_grad=True, device='cuda:0')
            q_opt = reference[:, AMPLoaderMorph.JOINT_POSE_START_IDX:AMPLoaderMorph.JOINT_POSE_END_IDX].to(torch.float32).clone().detach()
            # q_opt = (TOCABI_INIT_POS_TORCH.tile((reference_length, 1))).clone().detach()
            q_opt += 0.01*torch.rand_like(q_opt)
            q_opt = q_opt.requires_grad_(True)
            assert q_opt.shape == (reference_length, 12)
            optimizer = optim.Adam([q_opt], lr=1e-3)

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)
            print("SOLVING IK...")
            for i in range(n_iterations):
                optimizer.zero_grad()
                cost = cost_function(q_opt)[0]
                cost.backward()
                torch.nn.utils.clip_grad_norm_(q_opt, max_norm=1.0)
                optimizer.step()
                scheduler.step(cost)
                if i%100 == 0:
                    print(f"Iteration {i}, Cost: {cost.item()}")

                # Optionally, you can add some stopping criterion based on the change in cost or other conditions
                if optimizer.param_groups[0]['lr'] < 1e-6:
                    print("Stopping criterion met")
                    break
                assert ~torch.any(~torch.isfinite(q_opt)), f"Cannot solve IK! Iteration : {i}, Cost : {cost_function(q_opt)}"
            print("IK SOLVED!")

            temp = torch.zeros((q_opt.shape[0], 7), requires_grad=False, device=self.device, dtype=dtype)
            temp[:, :3] = to_torch(target_bodies['base_link']['info']['position'], device='cuda:0').tile((q_opt.shape[0], 1))
            temp[:, -1] = 1.        
            retarget_global_joint_pos, retarget_global_joint_quat, retarget_local_joint_pos, retarget_local_joint_quat = self.forward_kinematics(self.target, torch.cat((temp, q_opt), dim=-1), self.JOINT_MAPPING)
            ### FROM HEREON, CONSTRUCT THE RETARGET TRAJECTORY!!!###
            q_opt_prev, q_vel_opt = torch.zeros_like(q_opt), torch.zeros_like(q_opt)

            q_opt_prev[1:] = q_opt[:-1]
            q_vel_opt[1:] = (q_opt - q_opt_prev)[1:] * AMPLoaderMorph.HZ
            L_foot_pos_opt = retarget_local_joint_pos[FOOT_NAME[0]]
            R_foot_pos_opt = retarget_local_joint_pos[FOOT_NAME[1]]
            assert q_vel_opt.shape[0] == L_foot_pos_opt.shape[0]
            # time = torch.tensor([i for i in range(q_vel_opt.shape[0])], dtype=torch.float32, device='cuda:0').view(-1,1)
            # retarget_reference = torch.cat((retarget_global_joint_quat['virtual_joint'], q_opt, q_vel_opt, L_foot_pos_opt, R_foot_pos_opt), dim=-1)
            retarget_reference = torch.cat((self.root_ratio*reference[:, 0:1],retarget_global_joint_quat['virtual_joint'], self.root_ratio*source_local_base_lin_vel, source_local_base_ang_vel, q_opt, q_vel_opt, L_foot_pos_opt, R_foot_pos_opt), dim=-1)

            target_reference = torch.zeros((q_opt.shape[0], 3*13), requires_grad=False, device=self.device, dtype=dtype)
            for joint_name, _ in self.JOINT_MAPPING.items():
                target_reference[:, 3*(self.JOINT_MAPPING[joint_name]-6):3*(self.JOINT_MAPPING[joint_name]-5)] = retarget_global_joint_pos[joint_name]#target_global_joint_pos[joint_name] # retarget_global_joint_pos[joint_name]
            target_reference[:, 0:3] = retarget_global_joint_pos['virtual_joint'][:, 0:3]#target_global_joint_pos['virtual_joint'][:, 0:3] # retarget_global_joint_pos['virtual_joint'][:, 0:3]

            # return target_reference.detach()
            return retarget_reference.detach(), target_reference.detach()
        
        def retarget(self, source_motion_data, index, play=False, iterations=2000):
            source_reference, source_info = AMPLoaderMorph.process_data(source_motion_data, index)
            return self.retarget_tensor(source_reference, iterations, play)

        def forward_kinematics(self, model: dict, q: torch.Tensor, joint_map: dict, requires_grad = False):
            '''
            Computes the forward kinematics for a trajectory, given model = {bodies: dict, joints: dict}, q = torch.Tensor((traj_length, num_joints+7))
            joint_map is a dict that maps the joint names to their indices in q. For example, virtual joint = 0, hip yaw joint = 7, ...
            '''
            VIRTUAL_JOINT_DIM = 7 # pos and quat
            TRAJECTORY_LENGTH = q.shape[0]
            dtype = q.dtype
            bodies, joints = model['bodies'], model['joints']
            num_joints = q.shape[-1]-VIRTUAL_JOINT_DIM # q includes virtual joint. The rotation part is in quat.
            # The position and orientation of the frame attached to the joint. This will be the body frame for the child link.
            global_joint_pos = {}
            global_body_pos = {}
            global_joint_rot = {}
            global_body_rot = {}
            local_body_pos = {}
            local_body_rot = {}
            local_joint_pos = {}
            local_joint_rot = {}

            global_base_rot = None

            for body_name, body in bodies.items():
                parent_name = body['parent']
                joint_name = body['joint']
                body_info = body['info']
                if joint_name is not None:
                    joint_info = joints[joint_name]

                if body_name == "base_link": # Let's assume there is no virtual joint offset from base link frame origin
                    virtual_joint_value = q[:, joint_map[joint_name]:joint_map[joint_name]+VIRTUAL_JOINT_DIM]
                    global_joint_pos[joint_name] = virtual_joint_value[:, 0:3]
                    global_joint_rot[joint_name] = virtual_joint_value[:, 3:VIRTUAL_JOINT_DIM]
                    global_body_pos[body_name] = global_joint_pos[joint_name]
                    global_body_rot[body_name] = global_joint_rot[joint_name]
                    global_base_pos = global_joint_pos[joint_name]
                    global_base_rot = global_joint_rot[joint_name]
                    local_joint_pos[joint_name] = torch.zeros_like(global_joint_pos[joint_name])
                    local_joint_rot[joint_name] = torch.zeros_like(global_joint_rot[joint_name])
                    local_body_pos[body_name] = torch.zeros_like(global_base_pos)
                    local_body_rot[body_name] = torch.zeros_like(global_base_rot)

                    local_joint_rot[joint_name][:, -1] = 1. # Quaternion zero.
                elif joint_name in joint_map:
                    q_ = q[:, joint_map[joint_name]]
                    body_offset = to_torch(body_info['position'], dtype=dtype, device=self.device).tile(TRAJECTORY_LENGTH, 1)
                    joint_offset = to_torch(joint_info['position'], dtype=dtype, device=self.device).tile(TRAJECTORY_LENGTH, 1)
                    body_rot_offset = to_torch(body_info['quat'], dtype=dtype, device=self.device).tile(TRAJECTORY_LENGTH, 1)[:, [1,2,3,0]]
                    joint_rot_offset = to_torch(joint_info['quat'], dtype=dtype, device=self.device).tile(TRAJECTORY_LENGTH, 1)[:, [1,2,3,0]]
                    global_body_rot[body_name] = quat_mul(quat_mul(global_body_rot[parent_name], quat_from_angle_axis(q_, to_torch(joint_info['axis'], dtype=dtype, device=self.device))), body_rot_offset)
                    global_body_pos[body_name] = global_body_pos[parent_name] + quat_rotate(global_body_rot[parent_name], body_offset)
                    global_joint_pos[joint_name] = global_body_pos[body_name] + quat_rotate(global_body_rot[body_name], joint_offset)
                    global_joint_rot[joint_name] = quat_mul(global_body_rot[body_name], joint_rot_offset)

                    local_body_pos[joint_name] = quat_rotate_inverse(global_base_rot, (global_body_pos[body_name] - global_base_pos))
                    local_body_rot[joint_name] = quat_mul(quat_conjugate(global_base_rot), global_body_rot[body_name])
                    local_joint_pos[joint_name] = quat_rotate_inverse(global_base_rot, (global_joint_pos[joint_name] - global_base_pos))
                    local_joint_rot[joint_name] = quat_mul(quat_conjugate(global_base_rot), global_joint_rot[joint_name])

                elif joint_name is None:
                    # body is welded to parent
                    body_offset = to_torch(body_info['position'], dtype=dtype, device=self.device).tile(TRAJECTORY_LENGTH, 1)
                    body_rot_offset = to_torch(body_info['quat'], dtype=dtype, device=self.device).tile(TRAJECTORY_LENGTH, 1)[:, [1,2,3,0]]
                    global_body_rot[body_name] = quat_mul(global_body_rot[parent_name], body_rot_offset)
                    global_body_pos[body_name] = global_body_pos[parent_name] + quat_rotate(global_body_rot[parent_name], body_offset)
                    local_body_pos[joint_name] = quat_rotate_inverse(global_base_rot, (global_body_pos[body_name] - global_base_pos))
                    local_body_rot[joint_name] = quat_mul(quat_conjugate(global_base_rot), global_body_rot[body_name])
                else:
                    # upper body
                    pass
            return global_joint_pos, global_joint_rot, local_joint_pos, local_joint_rot




##################HELPER FUNCTIONS###################

def euler_to_rotation_matrix(euler_angles):
    """ Convert Euler angles (ZYX) to a rotation matrix. """
    B = euler_angles.shape[0]
    c = torch.cos(euler_angles)
    s = torch.sin(euler_angles)

    # Preallocate rotation matrix
    R = torch.zeros((B, 3, 3), dtype=torch.float16, device=euler_angles.device)

    # Fill in the entries
    R[:, 0, 0] = c[:, 1] * c[:, 0]
    R[:, 0, 1] = c[:, 1] * s[:, 0]
    R[:, 0, 2] = -s[:, 1]
    R[:, 1, 0] = s[:, 2] * s[:, 1] * c[:, 0] - c[:, 2] * s[:, 0]
    R[:, 1, 1] = s[:, 2] * s[:, 1] * s[:, 0] + c[:, 2] * c[:, 0]
    R[:, 1, 2] = s[:, 2] * c[:, 1]
    R[:, 2, 0] = c[:, 2] * s[:, 1] * c[:, 0] + s[:, 2] * s[:, 0]
    R[:, 2, 1] = c[:, 2] * s[:, 1] * s[:, 0] - s[:, 2] * c[:, 0]
    R[:, 2, 2] = c[:, 2] * c[:, 1]

    return R

def transform_vector(euler_angles, vectors):
    """ Transform vectors from frame A to B using a batch of Euler angles.
    
    Args:
    - euler_angles (numpy.array): Array of shape [B, 3] containing Euler angles.
    - vectors (numpy.array): Array of shape [B, 3] containing vectors in frame A.
    
    Returns:
    - transformed_vectors (numpy.array): Array of shape [B, 3] containing vectors in frame B.
    """
    # Convert inputs to torch tensors
    euler_angles = torch.tensor(euler_angles, dtype=torch.float16)
    vectors = torch.tensor(vectors, dtype=torch.float16)

    # Get rotation matrices from Euler angles
    R = euler_to_rotation_matrix(euler_angles)

    # Transform vectors
    transformed_vectors = torch.bmm(R, vectors.unsqueeze(-1)).squeeze(-1)

    # Convert back to numpy arrays
    return transformed_vectors.numpy()

# def quat_from_angle_axis(angle, axis: np.ndarray):
#     theta = (angle / 2).expand_dims(-1)
#     xyz = normalize(axis) * np.sin(theta)
#     w = np.cos(theta)
#     return quat_unit(np.concatenate((xyz, w), axis=-1))

def normalize(x: np.ndarray, eps: float = 1e-9):
    return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True).clip(eps, None)

def quat_unit(a):
    return normalize(a)

def denoise(data, window_length=5, polyorder=2):
    print("DENOISING!!")
    return savgol_filter(data, window_length=window_length, polyorder=polyorder, axis=0)


