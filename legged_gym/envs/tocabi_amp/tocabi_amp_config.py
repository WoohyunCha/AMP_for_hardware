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
import glob

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# MOTION_FILES = glob.glob('/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/motions_json/tocabi/*.json')
# MOTION_FILES = glob.glob('/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/motions_json/cmu/91/*.json')
MOTION_FILES = glob.glob('/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/motions_json/cmu/07/*.json')
# MOTION_FILES = glob.glob('/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/tocabi_data_scaled_1_0x.json')
REFERENCE_MODEL = '/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/data/raw/CMU_open/07/xml/07.xml'
# REFERENCE_MODEL = '/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/data/raw/CMU_open/91/xml/91.xml'
# REFERENCE_MODEL = None

class TOCABIAMPCfg( LeggedRobotCfg ):

    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        include_history_steps = 10  # Number of steps of history to include.
        skips = 2 # Number of steps to skip between steps in history
        num_observations = 48 # change 42
        num_privileged_obs = 48
        reference_state_initialization = True
        reference_state_initialization_prob = 1.
        amp_motion_files = MOTION_FILES
        episode_length_s = 20 # episode length in seconds
        num_actions = 12
        reference_model_file = REFERENCE_MODEL
        play = False

    class init_state( LeggedRobotCfg.init_state ):

        pos = [0.0, 0.0, 0.929869] # x,y,z [m]

        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]

        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]

        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        default_joint_angles = { # = target angles [rad] when action = 0.0

            'L_HipYaw_Joint': 0.0,

            'L_HipRoll_Joint': 0.0,

            'L_HipPitch_Joint': -0.28,

            'L_Knee_Joint': 0.6,

            'L_AnklePitch_Joint': -0.32,

            'L_AnkleRoll_Joint': 0.0,



            'R_HipYaw_Joint': 0.0,

            'R_HipRoll_Joint': 0.0,

            'R_HipPitch_Joint': -0.28,

            'R_Knee_Joint': 0.6,

            'R_AnklePitch_Joint': -0.32,

            'R_AnkleRoll_Joint': 0.0,



            "Waist1_Joint" : 0.,

            "Waist2_Joint" : 0.,

            "Upperbody_Joint" : 0.,

            "L_Shoulder1_Joint" : 0.3,

            "L_Shoulder2_Joint" : 0.174533,

            "L_Shoulder3_Joint" : 1.22173,

            "L_Armlink_Joint" : -1.27,

            "L_Elbow_Joint" : -1.57,

            "L_Forearm_Joint" : 0.,

            "L_Wrist1_Joint" : -1.,

            "L_Wrist2_Joint" : 0.,

            "Neck_Joint" : 0.,

            "Head_Joint" : 0.,

            "R_Shoulder1_Joint" : -0.3,

            "R_Shoulder2_Joint" : -0.174533,

            "R_Shoulder3_Joint" : -1.22173,

            "R_Armlink_Joint" : 1.27,

            "R_Elbow_Joint" : 1.57,

            "R_Forearm_Joint" : 0,

            "R_Wrist1_Joint" : 1,

            "R_Wrist2_Joint" : 0,

        }


    class control( LeggedRobotCfg.control ):

        # PD Drive parameters:

        control_type = 'T'

        stiffness = {"L_HipYaw_Joint": 2000.0, "L_HipRoll_Joint": 5000.0, "L_HipPitch_Joint": 4000.0,

            "L_Knee_Joint": 3700.0, "L_AnklePitch_Joint": 3200.0, "L_AnkleRoll_Joint": 3200.0,

            "R_HipYaw_Joint": 2000.0, "R_HipRoll_Joint": 5000.0, "R_HipPitch_Joint": 4000.0,

            "R_Knee_Joint": 3700.0, "R_AnklePitch_Joint": 3200.0, "R_AnkleRoll_Joint": 3200.0,



            "Waist1_Joint": 6000.0, "Waist2_Joint": 10000.0, "Upperbody_Joint": 10000.0,



            "L_Shoulder1_Joint": 400.0, "L_Shoulder2_Joint": 1000.0, "L_Shoulder3_Joint": 400.0, "L_Armlink_Joint": 400.0,

            "L_Elbow_Joint": 400.0, "L_Forearm_Joint": 400.0, "L_Wrist1_Joint": 100.0, "L_Wrist2_Joint": 100.0,



            "Neck_Joint": 100.0, "Head_Joint": 100.0,            



            "R_Shoulder1_Joint": 400.0, "R_Shoulder2_Joint": 1000.0, "R_Shoulder3_Joint": 400.0, "R_Armlink_Joint": 400.0,

            "R_Elbow_Joint": 400.0, "R_Forearm_Joint": 400.0, "R_Wrist1_Joint": 100.0, "R_Wrist2_Joint": 100.0}  # [N*m/rad]

        damping = {"L_HipYaw_Joint": 15.0, "L_HipRoll_Joint": 50.0, "L_HipPitch_Joint": 20.0,

            "L_Knee_Joint": 25.0, "L_AnklePitch_Joint": 24.0, "L_AnkleRoll_Joint": 24.0,

            "R_HipYaw_Joint": 15.0, "R_HipRoll_Joint": 50.0, "R_HipPitch_Joint": 20.0,

            "R_Knee_Joint": 25.0, "R_AnklePitch_Joint": 24.0, "R_AnkleRoll_Joint": 24.0,



            "Waist1_Joint": 200.0, "Waist2_Joint": 100.0, "Upperbody_Joint": 100.0,



            "L_Shoulder1_Joint": 10.0, "L_Shoulder2_Joint": 28.0, "L_Shoulder3_Joint": 10.0, "L_Armlink_Joint": 10.0,

            "L_Elbow_Joint": 10.0, "L_Forearm_Joint": 10.0, "L_Wrist1_Joint": 3.0, "L_Wrist2_Joint": 3.0,



            "Neck_Joint": 100.0, "Head_Joint": 100.0,            



            "R_Shoulder1_Joint": 10.0, "R_Shoulder2_Joint": 28.0, "R_Shoulder3_Joint": 10.0, "R_Armlink_Joint": 10.0,

            "R_Elbow_Joint": 10.0, "R_Forearm_Joint": 10.0, "R_Wrist1_Joint": 3.0, "R_Wrist2_Joint": 3.0}     # [N*m*s/rad]

        # action scale: target angle = actionScale * action + defaultAngle

        action_scale = 100.

        # decimation: Number of control action updates @ sim DT per policy DT

        decimation = 2 # TODO this was 6



    class terrain( LeggedRobotCfg.terrain ):

        mesh_type = 'plane'

        measure_heights = False



    class asset( LeggedRobotCfg.asset ):

        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/tocabi/urdf/tocabi.urdf' 

        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/tocabi/xml/dyros_tocabi.xml' 

        asset_is_mjcf = True
        name = "tocabi"
        foot_name = "Foot_Link"
        # foot_name = "AnkleRoll_Link"
        penalize_contacts_on = []
        # penalize_contacts_on = ['bolt_lower_leg_right_side', 'bolt_body', 'bolt_hip_fe_left_side', 'bolt_hip_fe_right_side', ' bolt_lower_leg_left_side', 'bolt_shoulder_fe_left_side', 'bolt_shoulder_fe_right_side', 'bolt_trunk', 'bolt_upper_leg_left_side', 'bolt_upper_leg_right_side']
        # terminate_after_contacts_on = ['base', 'Knee', 'Thigh', 'Head', 'Wrist']        
        terminate_after_contacts_on = ['base', 'Knee', 'Thigh', 'Head', 'Wrist', 'Foot_Redundant']        
        termination_height = (0.7, 1.0)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        # fix_base_link = True
          
    class domain_rand:
        randomize_friction = False
        friction_range = [0.8, 1.2]
        randomize_base_mass = False
        added_mass_range = [-5., 5.]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 0.3
        randomize_gains = False
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]
        randomize_torque = False
        torque_constant = [0.98, 1.02]
        randomize_joints = False
        damping_range = [0., 2.]
        armature_range = [0.8, 1.2]

    class noise:
        add_noise = True
        noise_level = .0003
        noise_dist = 'uniform' # gaussian
        class noise_scales:
            dof_pos = 1
            dof_vel = 250
            lin_vel = 100
            ang_vel = 100
            gravity = 1
            height_measurements = 10

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        tracking_sigma = 0.1
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = 0.0
            tracking_lin_vel = 50
            tracking_ang_vel = 25
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 0.0
            torques = 0.0
            dof_vel = 0.0
            dof_acc = 0.0
            base_height = 0.0 
            feet_air_time =  0.0
            collision = 0.0
            feet_stumble = 0.0 
            action_rate = 0.0
            stand_still = 0.0
            dof_pos_limits = 0.0

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.
        normalize_observation = True # True means we keep running estimates of mean and variance to normalize the observation vector

    class commands:
        curriculum = False
        max_curriculum = 2.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [0., 1.] # min max [m/s]
            lin_vel_y = [-0., 0.]   # min max [m/s]
            ang_vel_yaw = [-0., 0.]    # min max [rad/s]
            heading = [-0., 0.]



    class sim:

        dt =  0.002

        substeps = 1

        gravity = [0., 0. ,-9.81]  # [m/s^2]

        up_axis = 1  # 0 is y, 1 is z



        class physx:

            num_threads = 10

            solver_type = 1  # 0: pgs, 1: tgs

            num_position_iterations = 4

            num_velocity_iterations = 0

            contact_offset = 0.01  # [m]

            rest_offset = 0.0   # [m]

            bounce_threshold_velocity = 0.5 #0.5 [m/s]

            max_depenetration_velocity = 1.0

            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more

            default_buffer_size_multiplier = 5

            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class TOCABIAMPCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'AMPOnPolicyRunner'
    seed = 1

    class policy:

        init_noise_std = 1.0

        actor_hidden_dims = [512, 512]

        critic_hidden_dims = [512, 512]

        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        # only for 'ActorCriticRecurrent':

        # rnn_type = 'lstm'

        # rnn_hidden_size = 512

        # rnn_num_layers = 1

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        value_loss_coef = .5
        entropy_coef = 0.01
        amp_replay_buffer_size = 100000
        num_learning_epochs = 5
        num_mini_batches = 4
        disc_coef = 5
        bounds_loss_coef = 10
        disc_grad_pen = 0.1
        learning_rate = 2.e-5

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'tocabi_amp' # should be the same as 'env' in env.py and env_config.py 
        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 20000 # number of policy updates

        amp_reward_coef = 2.0
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.5
        amp_discr_hidden_dims = [256, 256]
        amp_reference_model = REFERENCE_MODEL

        min_normalized_std = [0.05, 0.02, 0.05] * 4
        LOG_WANDB = True
        wgan = False