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
from rsl_rl.datasets.mocap_motions.data.raw.CMU_open.CMU_parser import MotionRetarget
from scipy.signal import savgol_filter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'mocap_motions', 'AMP_trajectories')
MJCF_file = '/home/cha/isaac_ws/AMP_for_hardware/resources/robots/tocabi/xml/dyros_tocabi.xml'



class AMPLoader:
    FRAME_TIME = 0
    MODEL_DOF = 33
    REFERENCE_START_INDEX = None
    REFERENCE_END_INDEX = None


    POS_SIZE = 3
    JOINT_POS_SIZE = 12
    LINEAR_VEL_SIZE = 0
    ANGULAR_VEL_SIZE = 0
    JOINT_VEL_SIZE = 12
    ROOT_POS_SIZE = 0 # base height 
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
                # motion_data = self.reorder_from_pybullet_to_isaac(motion_data) # For A1 only

                # # Normalize and standardize quaternions.
                # for f_i in range(motion_data.shape[0]):
                #     root_rot = AMPLoader.get_root_rot(motion_data[f_i])
                #     root_rot = pose3d.QuaternionNormalize(root_rot)
                #     root_rot = motion_util.standardize_quaternion(root_rot)
                #     motion_data[
                #         f_i,
                #         AMPLoader.POS_SIZE:
                #             (AMPLoader.POS_SIZE +
                #              AMPLoader.ROT_SIZE)] = root_rot
                if (model_file == target_model_file or model_file == ''): 
                    print("No retargetting motion")
                    processed_data_full, info = self.process_data(motion_data, i)
                    processed_data_joints = processed_data_full[:, self.JOINT_POSE_START_IDX:self.JOINT_VEL_END_IDX]

                else:
                    print("RETARGET REFERENCE MOTIONS")
                    print("Target file : ", target_model_file)
                    print("Target file : ", info['xml'])
                    motion_retarget = MotionRetarget(source_model_path=model_file, target_model_path=target_model_file) # model_file is the source
                    processed_data_full, _ = motion_retarget.retarget(motion_data, play, iterations)
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
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
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
        # all_frame_pos_starts = torch.zeros(len(traj_idxs), AMPLoader.ROOT_POS_SIZE, device=self.device)
        # all_frame_pos_ends = torch.zeros(len(traj_idxs), AMPLoader.ROOT_POS_SIZE, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), AMPLoader.ROOT_ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), AMPLoader.ROOT_ROT_SIZE, device=self.device)
        # all_frame_linvel_starts = torch.zeros(len(traj_idxs), AMPLoader.LINEAR_VEL_SIZE, device=self.device)
        # all_frame_linvel_ends = torch.zeros(len(traj_idxs), AMPLoader.LINEAR_VEL_SIZE, device=self.device)
        # all_frame_angvel_starts = torch.zeros(len(traj_idxs), AMPLoader.ANGULAR_VEL_SIZE, device=self.device)
        # all_frame_angvel_ends = torch.zeros(len(traj_idxs), AMPLoader.ANGULAR_VEL_SIZE, device=self.device)
        all_frame_amp_starts = torch.zeros(len(traj_idxs), AMPLoader.JOINT_VEL_END_IDX - AMPLoader.JOINT_POSE_START_IDX, device=self.device)
        all_frame_amp_ends = torch.zeros(len(traj_idxs),  AMPLoader.JOINT_VEL_END_IDX - AMPLoader.JOINT_POSE_START_IDX, device=self.device)
        all_frame_foot_pos_starts = torch.zeros(len(traj_idxs), AMPLoader.FOOT_POS_END_IDX - AMPLoader.FOOT_POS_START_IDX, device=self.device)
        all_frame_foot_pos_ends = torch.zeros(len(traj_idxs), AMPLoader.FOOT_POS_END_IDX - AMPLoader.FOOT_POS_START_IDX, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            # all_frame_pos_starts[traj_mask] = AMPLoader.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            # all_frame_pos_ends[traj_mask] = AMPLoader.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = AMPLoader.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = AMPLoader.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            # all_frame_linvel_starts[traj_mask] = AMPLoader.get_linear_vel_batch(trajectory[idx_high[traj_mask]])
            # all_frame_linvel_ends[traj_mask] = AMPLoader.get_linear_vel_batch(trajectory[idx_low[traj_mask]])
            # all_frame_angvel_starts[traj_mask] = AMPLoader.get_angular_vel_batch(trajectory[idx_high[traj_mask]])
            # all_frame_angvel_ends[traj_mask] = AMPLoader.get_angular_vel_batch(trajectory[idx_low[traj_mask]])
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][:, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_VEL_END_IDX]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][:, AMPLoader.JOINT_POSE_START_IDX:AMPLoader.JOINT_VEL_END_IDX]
            all_frame_foot_pos_starts[traj_mask] = AMPLoader.get_foot_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_foot_pos_ends[traj_mask] = AMPLoader.get_foot_pos_batch(trajectory[idx_high[traj_mask]])

        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32, requires_grad=False).unsqueeze(-1)

        # pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend) 
        rot_blend = self.slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        # lin_blend = self.slerp(all_frame_linvel_starts, all_frame_linvel_ends, blend)
        # ang_blend = self.slerp(all_frame_angvel_starts, all_frame_angvel_ends, blend)
        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        foot_pos_blend = self.slerp(all_frame_foot_pos_starts, all_frame_foot_pos_ends, blend)

        # ret = torch.cat([pos_blend, rot_blend, lin_blend, ang_blend, amp_blend, foot_pos_blend], dim=-1) 
        # ret = torch.cat([pos_blend, rot_blend, amp_blend, foot_pos_blend], dim=-1) 
        ret = torch.cat([rot_blend, amp_blend, foot_pos_blend], dim=-1) 
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

        # root_pos0, root_pos1 = AMPLoader.get_root_pos(frame0), AMPLoader.get_root_pos(frame1)
        root_rot0, root_rot1 = AMPLoader.get_root_rot(frame0), AMPLoader.get_root_rot(frame1)
        # linear_vel_0, linear_vel_1 = AMPLoader.get_linear_vel(frame0), AMPLoader.get_linear_vel(frame1)
        # angular_vel_0, angular_vel_1 = AMPLoader.get_angular_vel(frame0), AMPLoader.get_angular_vel(frame1)
        joints0, joints1 = AMPLoader.get_joint_pose(frame0), AMPLoader.get_joint_pose(frame1)
        joint_vel_0, joint_vel_1 = AMPLoader.get_joint_vel(frame0), AMPLoader.get_joint_vel(frame1)
        foot_pos0, foot_pos1 = AMPLoader.get_foot_pos(frame0), AMPLoader.get_foot_pos(frame1)
        
        # blend_root_pos = self.slerp(root_pos0, root_pos1, blend) 
        blend_root_rot = self.slerp(root_rot0, root_rot1, blend)
        # blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        # blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)
        blend_foot_pos = self.slerp(foot_pos0, foot_pos1, blend)

        # ret = torch.cat([
        #     blend_root_pos, blend_root_rot, blend_linear_vel, blend_angular_vel, blend_joints, blend_joints_vel,
        #     blend_foot_pos
        #     ], dim=-1)
        # ret = torch.cat([ 
        #     blend_root_pos, blend_root_rot, blend_joints, blend_joints_vel,
        #     blend_foot_pos
        #     ], dim=-1)
        ret = torch.cat([
            blend_root_rot, blend_joints, blend_joints_vel,
            blend_foot_pos
            ], dim=-1)

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
            yield s, s_next

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.OBSERVATION_DIM

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    # def get_root_pos(pose):
    #     return pose[AMPLoader.ROOT_POS_START_IDX:AMPLoader.ROOT_POS_END_IDX]

    # def get_root_pos_batch(poses):
    #     return poses[:, AMPLoader.ROOT_POS_START_IDX:AMPLoader.ROOT_POS_END_IDX]

    def get_root_rot(pose):
        return pose[AMPLoader.ROOT_ROT_START_IDX:AMPLoader.ROOT_ROT_END_IDX]

    def get_root_rot_batch(poses):
        return poses[:, AMPLoader.ROOT_ROT_START_IDX:AMPLoader.ROOT_ROT_END_IDX]

    # def get_linear_vel(pose):
    #     return pose[AMPLoader.LINEAR_VEL_START_IDX:AMPLoader.LINEAR_VEL_END_IDX]
    
    # def get_linear_vel_batch(poses):
    #     return poses[:, AMPLoader.LINEAR_VEL_START_IDX:AMPLoader.LINEAR_VEL_END_IDX]

    # def get_angular_vel(pose):
    #     return pose[AMPLoader.ANGULAR_VEL_START_IDX:AMPLoader.ANGULAR_VEL_END_IDX]  

    # def get_angular_vel_batch(poses):
    #     return poses[:, AMPLoader.ANGULAR_VEL_START_IDX:AMPLoader.ANGULAR_VEL_END_IDX]
    
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

    def process_data(self, traj: np.ndarray, index: int):# Process raw data from Mujoco
        traj = denoise(traj, window_length=int(0.2*AMPLoader.HZ), polyorder=2)
        time = traj[AMPLoader.REFERENCE_START_INDEX:AMPLoader.REFERENCE_END_INDEX,0]
        pelv_pos = traj[AMPLoader.REFERENCE_START_INDEX:AMPLoader.REFERENCE_END_INDEX, 1:4]
        pelv_rpy = traj[AMPLoader.REFERENCE_START_INDEX:AMPLoader.REFERENCE_END_INDEX, 4:7] # roll, pitch, yaw order
        pelv_vel = traj[AMPLoader.REFERENCE_START_INDEX:AMPLoader.REFERENCE_END_INDEX, 7:13]
        q_pos = traj[AMPLoader.REFERENCE_START_INDEX:AMPLoader.REFERENCE_END_INDEX, 13:13+AMPLoader.JOINT_POS_SIZE]
        start = 13+AMPLoader.MODEL_DOF
        end = start + self.JOINT_VEL_SIZE
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
        assert end == 1+3+3+6+self.MODEL_DOF+self.MODEL_DOF+3+3+3+3, f"process data, shape mismatch {end}"

        # torch versions of data
        pelv_pos_torch = to_torch(pelv_pos)
        pelv_vel_torch = to_torch(pelv_vel)
        q_pos_torch = to_torch(q_pos)
        q_vel_torch = to_torch(q_vel)
        L_foot_pos_torch = to_torch(L_foot_pos)
        L_foot_rpy_torch = to_torch(L_foot_rpy)
        R_foot_pos_torch = to_torch(R_foot_pos)
        R_foot_rpy_torch = to_torch(R_foot_rpy)

        # Process raw data
        pelvis_yaw =pelv_rpy[:, 2]
        pelvis_pitch = pelv_rpy[:, 1]
        pelvis_roll = pelv_rpy[:, 0]
        pelvis_quat = quat_from_euler_xyz(to_torch(pelvis_roll), to_torch(pelvis_pitch), to_torch(pelvis_yaw)) # tensor
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
        # ret = torch.concat((base_height.unsqueeze(-1), pelvis_quat, base_lin_vel, base_ang_vel, q_pos_torch, q_vel_torch, L_foot_pos_base_torch, R_foot_pos_base_torch), dim=-1) 
        # ret = torch.concat((base_height.unsqueeze(-1), pelvis_quat, q_pos_torch, q_vel_torch, L_foot_pos_base_torch, R_foot_pos_base_torch), dim=-1) 
        ret = torch.concat((pelvis_quat, q_pos_torch, q_vel_torch, L_foot_pos_base_torch, R_foot_pos_base_torch), dim=-1) 
        info = torch.concat((base_height.unsqueeze(-1), base_lin_vel, base_ang_vel, L_foot_quat_base_torch, R_foot_quat_base_torch), dim=-1)
        with open(os.path.join(PROCESSED_DATA_DIR, 'processed_data_'+str(index)+'.txt'), "w") as file:
            for line in ret.cpu().numpy().tolist():
                file.write(' '.join(map(str, line)) + '\n')
            print(f"Processed data {index} written to txt file at "+PROCESSED_DATA_DIR)
        return ret, info
    

##################HELPER FUNCTIONS###################

def euler_to_rotation_matrix(euler_angles):
    """ Convert Euler angles (ZYX) to a rotation matrix. """
    B = euler_angles.shape[0]
    c = torch.cos(euler_angles)
    s = torch.sin(euler_angles)

    # Preallocate rotation matrix
    R = torch.zeros((B, 3, 3), dtype=torch.float32, device=euler_angles.device)

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
    euler_angles = torch.tensor(euler_angles, dtype=torch.float32)
    vectors = torch.tensor(vectors, dtype=torch.float32)

    # Get rotation matrices from Euler angles
    R = euler_to_rotation_matrix(euler_angles)

    # Transform vectors
    transformed_vectors = torch.bmm(R, vectors.unsqueeze(-1)).squeeze(-1)

    # Convert back to numpy arrays
    return transformed_vectors.numpy()



##########HELPER###########
def quat_from_angle_axis(angle, axis: np.ndarray):
    theta = (angle / 2).expand_dims(-1)
    xyz = normalize(axis) * np.sin(theta)
    w = np.cos(theta)
    return quat_unit(np.concatenate((xyz, w), axis=-1))

def normalize(x: np.ndarray, eps: float = 1e-9):
    return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True).clip(eps, None)

def quat_unit(a):
    return normalize(a)

def denoise(data, window_length=5, polyorder=2):
    print("DENOISING!!")
    return savgol_filter(data, window_length=window_length, polyorder=polyorder, axis=0)