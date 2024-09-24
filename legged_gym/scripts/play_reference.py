# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import numpy as np
import torch
from rsl_rl.datasets.motion_loader import AMPLoader, AMPLoaderMorph
import glob

# REFERENCE_MOTION_FILE =  glob.glob('/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/motions_json/cmu/07/07_slow_walk_2.json')
# REFERENCE_MOTION_FILE =  glob.glob('/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/motions_json/cmu/91/91_straight_walk.json')
# REFERENCE_MOTION_FILE =  glob.glob('/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/motions_json/cmu/69/69_walk_forward_01.json')
# REFERENCE_MOTION_FILE =  glob.glob('/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/motions_json/tocabi/tocabi_data_scaled_1_0x.json')
# REFERENCE_HZ = 120
# REFERENCE_HZ = 2000
# REFERENCE_MODEL_FILE = '/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/data/raw/CMU_open/07/xml/07.xml'
# REFERENCE_MODEL_FILE = '/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/data/raw/CMU_open/91/xml/91.xml'
# REFERENCE_MODEL_FILE = '/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/data/raw/CMU_open/69/xml/69.xml'
REFERENCE_DICT = {
    '/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/motions_json/tocabi/tocabi_data_scaled_1_0x.json': {
        'xml': '/home/cha/isaac_ws/AMP_for_hardware/resources/robots/tocabi/xml/dyros_tocabi.xml',
        'hz' : 2000,
        'start_time' : 4.5,
        'end_time' : 8.1,
        'model_dof' : 33,
        'weight' : 1.
    },
}
REFERENCE_JSON, REFERENCE_INFO = next(iter(REFERENCE_DICT.items()))
REFERENCE_MODEL_FILE = REFERENCE_INFO['xml']
REFERENCE_HZ = REFERENCE_INFO['hz']
RENDER_HZ = 100

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.asset.num_morphologies = 1
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 2)
    env_cfg.env.amp_motion_files = REFERENCE_DICT
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.sim.dt = 1/REFERENCE_HZ
    env_cfg.asset.fix_base_link = True
    if args.speed is not None:
        env_cfg.commands.num_commands = 4
        env_cfg.commands.heading_command = True        
        env_cfg.commands.ranges.lin_vel_x = [args.speed, args.speed]
        env_cfg.commands.ranges.lin_vel_y = [0., 0.]
        env_cfg.commands.ranges.heading = [0.,0.]
    train_cfg.runner.amp_num_preload_transitions = 100

    # prepare environment

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    source_asset_path = env_cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    source_asset_root = os.path.dirname(source_asset_path)
    source_asset_file = os.path.basename(source_asset_path)
    target_asset_file = os.path.splitext(source_asset_file)[0] + f'_randomized_0.xml'    # env.dt = 1/REFERENCE_HZ
    target_asset_path = os.path.join(source_asset_root, target_asset_file)
    # motion_tensor = AMPLoaderMorph('cuda:0', env.dt, reference_dict=REFERENCE_DICT, target_model_file=target_asset_path).trajectories[0].to(torch.float32)

    traj_dict = torch.load('/home/cha/isaac_ws/AMP_for_hardware/legged_gym/scripts/trajectory.pt')
    traj = traj_dict['trajectory'] # shape (T, M, d)
    morph = traj_dict['morphology']

    motion_tensor = traj[:, 0, :].to(torch.float32)


    # # load policy
    # train_cfg.runner.resume = True
    # train_cfg.runner.LOG_WANDB = False
    # ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, play=True)
    # policy = ppo_runner.get_inference_policy(device=env.device)
    # ppo_runner.env.set_normalizer_eval()    
    # _, _ = env.reset()
    # obs = env.get_observations()

    # logger = Logger(env.dt)
    # robot_index = 0 # which robot is used for logging
    # joint_index = 1 # which joint is used for logging
    # start_state_log = np.ceil(2. / env.dt) 
    # stop_state_log = np.ceil(6. / env.dt) # number of steps before plotting states
    # stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    # camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    # camera_vel = np.array([1., 1., 0.])
    # camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    # img_idx = 0
    frames = motion_tensor#[REFERENCE_START_INDEX:REFERENCE_END_INDEX]
    for _ in range(100):
        for i in range(frames.shape[0]):
            if i % int(REFERENCE_HZ/RENDER_HZ) == 0:
                frame = frames[i]
                env.step_forced(frame)

if __name__ == '__main__':
    # EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
