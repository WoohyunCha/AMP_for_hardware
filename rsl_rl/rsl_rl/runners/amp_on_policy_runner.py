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

import time
import os
from collections import deque
import statistics

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import AMPPPO, PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv
from rsl_rl.algorithms.amp_discriminator import AMPDiscriminator
from rsl_rl.datasets.motion_loader import AMPLoader
from rsl_rl.utils.utils import Normalizer
from legged_gym.utils.helpers import class_to_dict

import wandb
from datetime import datetime

LEGGED_GYM_ROOT = '/home/cha/isaac_ws/AMP_for_hardware/legged_gym'
LEGGED_GYM_ENVS = '/home/cha/isaac_ws/AMP_for_hardware/legged_gym/envs'

class AMPOnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):


        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        if self.env.include_history_steps is not None:
            num_actor_obs = self.env.num_obs * self.env.include_history_steps
        else:
            num_actor_obs = self.env.num_obs
        actor_critic: ActorCritic = actor_critic_class( num_actor_obs=num_actor_obs,
                                                        num_critic_obs=num_critic_obs,
                                                        num_actions=self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)

        amp_data = AMPLoader(
            device, time_between_frames=self.env.dt, preload_transitions=True,
            num_preload_transitions=train_cfg['runner']['amp_num_preload_transitions'], 
            motion_files=self.cfg["amp_motion_files"])
        print("AMP data observation data size : ", amp_data.observation_dim)
        amp_normalizer = Normalizer(amp_data.observation_dim) # batchnorm. Updates its running averages (mean, std) of observations from batches of observations, and normalizes observations using them
        discriminator = AMPDiscriminator( # Discriminator computes the reward and gradient penalty loss.
            amp_data.observation_dim * 2,
            train_cfg['runner']['amp_reward_coef'],
            train_cfg['runner']['amp_discr_hidden_dims'], device,
            train_cfg['runner']['amp_task_reward_lerp']).to(self.device)

        # self.discr: AMPDiscriminator = AMPDiscriminator()
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        min_std = (
            torch.tensor(self.cfg["min_normalized_std"], device=self.device) *
            (torch.abs(self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0]))[:self.env.num_actions])
        self.alg: PPO = alg_class(actor_critic, discriminator, amp_data, amp_normalizer, device=self.device, min_std=min_std, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [num_actor_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.LOG_WANDB = self.cfg['LOG_WANDB']

        # WANDB INIT
        if self.LOG_WANDB:
            wandb.init(project="AMP_tocabi")
            experiment_name = train_cfg['runner']['experiment_name']
            wandb.run.name = experiment_name + '_' + datetime.now().strftime("%Y%m%d_%H%M%S")
            wandb.run.save()

            args = class_to_dict(train_cfg)
            wandb.config.update(args)

            wandb.save(os.path.join(LEGGED_GYM_ENVS, experiment_name,experiment_name+'_config.py'),policy="now")

            wandb.save(os.path.join(LEGGED_GYM_ENVS, experiment_name,experiment_name+'.py'), policy="now")

            wandb.save('/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/runners/amp_on_policy_runner.py', policy="now")

            wandb.save('/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/motion_loader.py', policy="now")

            wandb.save('/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/algorithms/amp_discriminator.py', policy="now")

            wandb.save('/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/algorithms/amp_ppo.py', policy="now")

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        if self.env.normalizer_obs is not None:
            self.env.normalizer_obs.train()
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        amp_obs = self.env.get_amp_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, amp_obs = obs.to(self.device), critic_obs.to(self.device), amp_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)
        self.alg.discriminator.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        amprewbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, amp_obs)
                    obs, privileged_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states = self.env.step(actions)
                    next_amp_obs = self.env.get_amp_observations()

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, next_amp_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), next_amp_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # Account for terminal states.
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    rewards, amp_rewards, _ = self.alg.discriminator.predict_amp_reward(
                        amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer)
                    amp_obs = torch.clone(next_amp_obs)
                    self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        amprewbuffer.extend(amp_rewards.cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))

            wandb_dict = {

                "dt" : self.env.dt,

                "Train/mean_policy_pred": mean_policy_pred,

                "Train/mean_expert_pred": mean_expert_pred,

            }

            if it%10 == 0:

                self.log_wandb(wandb_dict, locals())

            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
        self.save_wandb(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP', locs['mean_amp_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP_grad', locs['mean_grad_pen_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                          f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                          f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                          f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'discriminator_state_dict': self.alg.discriminator.state_dict(),
            'amp_normalizer': self.alg.amp_normalizer,
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)
        if self.env.normalizer_obs is not None:
            torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'discriminator_state_dict': self.alg.discriminator.state_dict(),
            'amp_normalizer': self.alg.amp_normalizer,
            'obs_normalizer_state_dict': self.env.normalizer_obs.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'])
        self.alg.amp_normalizer = loaded_dict['amp_normalizer']
        if self.env.normalizer_obs is not None:
            self.env.normalizer_obs.load_state_dict(loaded_dict['obs_normalizer_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    
    def get_inference_normalizer(self, device=None):
        self.env.normalizer_obs.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.env.normalizer_obs.to(device)
        return self.env.normalizer_obs

    def log_wandb(self, d, locs):
        if self.LOG_WANDB:
            wandb_dict = dict()
            wandb_dict = {**wandb_dict, **d}
            wandb_dict['n_updates'] = locs['it']
            wandb_dict['Loss/value_function'] = locs['mean_value_loss']
            wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
            wandb_dict['Loss/AMP'] = locs['mean_amp_loss']
            wandb_dict['Loss/AMP_grad'] = locs['mean_grad_pen_loss']
            wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
            if locs['ep_infos']:
                for key in locs['ep_infos'][0]:
                    infotensor = torch.tensor([], device=self.device)
                    for ep_info in locs['ep_infos']:
                        # handle scalar and zero dimensional tensor infos
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.Tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                    value = torch.mean(infotensor)
                    wandb_dict['Train/Mean_episode_' + key] = value.item()
            if len(locs['rewbuffer']) > 0:
                wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
                wandb_dict['Train_AMP/Mean_episode_style_reward'] = statistics.mean(locs['amprewbuffer'])                
                wandb_dict['Train/mean_episode_length_t'] = statistics.mean(locs['lenbuffer']) * wandb_dict['dt']
                wandb.log(wandb_dict)

    def save_wandb(self, model_path):
        if self.LOG_WANDB:
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(model_path)
            wandb.run.log_artifact(artifact)
            wandb.run.finish()

# # SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# # SPDX-License-Identifier: BSD-3-Clause
# # 
# # Redistribution and use in source and binary forms, with or without
# # modification, are permitted provided that the following conditions are met:
# #
# # 1. Redistributions of source code must retain the above copyright notice, this
# # list of conditions and the following disclaimer.
# #
# # 2. Redistributions in binary form must reproduce the above copyright notice,
# # this list of conditions and the following disclaimer in the documentation
# # and/or other materials provided with the distribution.
# #
# # 3. Neither the name of the copyright holder nor the names of its
# # contributors may be used to endorse or promote products derived from
# # this software without specific prior written permission.
# #
# # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# #
# # Copyright (c) 2021 ETH Zurich, Nikita Rudin

# import time
# import os
# from collections import deque
# import statistics

# import numpy as np
# from torch.utils.tensorboard import SummaryWriter
# import torch

# from rsl_rl.algorithms import AMPPPO, PPO
# from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
# from rsl_rl.env import VecEnv
# from rsl_rl.algorithms.amp_discriminator import AMPDiscriminator, AMPCritic
# from rsl_rl.datasets.motion_loader import AMPLoader
# from rsl_rl.utils.utils import Normalizer
# from legged_gym.utils.helpers import class_to_dict
# from legged_gym import LEGGED_GYM_ROOT_DIR

# import wandb
# from datetime import datetime

# LEGGED_GYM_ROOT = '/home/cha/isaac_ws/AMP_for_hardware/legged_gym'
# LEGGED_GYM_ENVS = '/home/cha/isaac_ws/AMP_for_hardware/legged_gym/envs'
# RSLRL_ROOT = '/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl'

# class AMPOnPolicyRunner:

#     def __init__(self,
#                  env: VecEnv,
#                  train_cfg,
#                  log_dir=None,
#                  device='cpu'):


#         self.cfg=train_cfg["runner"]
#         self.alg_cfg = train_cfg["algorithm"]
#         self.policy_cfg = train_cfg["policy"]
#         self.device = device
#         self.env = env
#         if self.env.num_privileged_obs is not None:
#             num_critic_obs = self.env.num_privileged_obs 
#         else:
#             num_critic_obs = self.env.num_obs
#         actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
#         if self.env.include_history_steps is not None:
#             num_actor_obs = self.env.num_obs * self.env.include_history_steps
#         else:
#             num_actor_obs = self.env.num_obs
#         actor_critic: ActorCritic = actor_critic_class( num_actor_obs=num_actor_obs,
#                                                         num_critic_obs=num_critic_obs,
#                                                         num_actions=self.env.num_actions,
#                                                         **self.policy_cfg).to(self.device)
#         print("model file : ", self.env.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))

#         amp_data = AMPLoader(
#             device, time_between_frames=self.env.dt, preload_transitions=True,
#             num_preload_transitions=train_cfg['runner']['amp_num_preload_transitions'], 
#             motion_files=self.cfg["amp_motion_files"],
#             model_file=self.env.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
#             )
#         print("AMP data observation data size : ", amp_data.observation_dim)
#         amp_normalizer = Normalizer(amp_data.observation_dim) # batchnorm. Updates its running averages (mean, std) of observations from batches of observations, and normalizes observations using them
#         if False: # self.cfg['wgan']:
#             print("WGAN is used!!!")
#             discriminator = AMPCritic(
#                 amp_data.observation_dim * 2,
#                 train_cfg['runner']['amp_reward_coef'],
#                 train_cfg['runner']['amp_discr_hidden_dims'], device,
#                 train_cfg['runner']['amp_task_reward_lerp']).to(self.device)
#         else:
#             print("LSGAN is used!!!")
#             discriminator = AMPDiscriminator( # Discriminator computes the reward and gradient penalty loss.
#                 amp_data.observation_dim * 2,
#                 train_cfg['runner']['amp_reward_coef'],
#                 train_cfg['runner']['amp_discr_hidden_dims'], device,
#                 train_cfg['runner']['amp_task_reward_lerp']).to(self.device)

#         # self.discr: AMPDiscriminator = AMPDiscriminator()
#         alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
#         min_std = (
#             torch.tensor(self.cfg["min_normalized_std"], device=self.device) *
#             (torch.abs(self.env.dof_pos_limits[:, 1] - self.env.dof_pos_limits[:, 0]))[:self.env.num_actions])
#         self.alg: PPO = alg_class(actor_critic, discriminator, amp_data, amp_normalizer, device=self.device, min_std=min_std, **self.alg_cfg)
#         self.num_steps_per_env = self.cfg["num_steps_per_env"]
#         self.save_interval = self.cfg["save_interval"]

#         # init storage and model
#         self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [num_actor_obs], [self.env.num_privileged_obs], [self.env.num_actions])

#         # Log
#         self.log_dir = log_dir
#         self.writer = None
#         self.tot_timesteps = 0
#         self.tot_time = 0
#         self.current_learning_iteration = 0
#         self.LOG_WANDB = self.cfg['LOG_WANDB']

#         # WANDB INIT
#         if self.LOG_WANDB:
#             wandb.init(project="AMP_tocabi")
#             experiment_name = train_cfg['runner']['experiment_name']
#             wandb.run.name = experiment_name + '_' + datetime.now().strftime("%Y%m%d_%H%M%S")
#             wandb.run.save()

#             args = class_to_dict(train_cfg)
#             wandb.config.update(args)
#             wandb.save(os.path.join(LEGGED_GYM_ENVS, experiment_name,experiment_name+'_config.py'),policy="now")
#             wandb.save(os.path.join(LEGGED_GYM_ENVS, experiment_name,experiment_name+'.py'), policy="now")
#             wandb.save('/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/runners/amp_on_policy_runner.py', policy="now")
#             wandb.save('/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/motion_loader.py', policy="now")
#             wandb.save('/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/algorithms/amp_discriminator.py', policy="now")
#             wandb.save('/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/algorithms/amp_ppo.py', policy="now")


#         _, _ = self.env.reset()
    
#     def learn(self, num_learning_iterations, init_at_random_ep_len=False):
#         # initialize writer
#         if self.log_dir is not None and self.writer is None:
#             self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
#         if init_at_random_ep_len:
#             self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
#         obs = self.env.get_observations()
#         privileged_obs = self.env.get_privileged_observations()
#         amp_obs = self.env.get_amp_observations()
#         critic_obs = privileged_obs if privileged_obs is not None else obs
#         obs, critic_obs, amp_obs = obs.to(self.device), critic_obs.to(self.device), amp_obs.to(self.device)
#         self.alg.actor_critic.train() # switch to train mode (for dropout for example)
#         self.alg.discriminator.train()

#         ep_infos = []
#         rewbuffer = deque(maxlen=100)
#         lenbuffer = deque(maxlen=100)
#         cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
#         cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

#         tot_iter = self.current_learning_iteration + num_learning_iterations
#         for it in range(self.current_learning_iteration, tot_iter):
#             start = time.time()
#             # Rollout
#             with torch.inference_mode():
#                 for i in range(self.num_steps_per_env):
#                     actions = self.alg.act(obs, critic_obs, amp_obs)
#                     obs, privileged_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states = self.env.step(actions)
#                     next_amp_obs = self.env.get_amp_observations()

#                     critic_obs = privileged_obs if privileged_obs is not None else obs
#                     obs, critic_obs, next_amp_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), next_amp_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

#                     # Account for terminal states.
#                     next_amp_obs_with_term = torch.clone(next_amp_obs)
#                     next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

#                     rewards, d = self.alg.discriminator.predict_amp_reward(
#                         amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer)
#                     amp_obs = torch.clone(next_amp_obs)
                    
#                     self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)
                    
#                     if self.log_dir is not None:
#                         # Book keeping
#                         if 'episode' in infos:
#                             ep_infos.append(infos['episode'])
#                         cur_reward_sum += rewards
#                         cur_episode_length += 1
#                         new_ids = (dones > 0).nonzero(as_tuple=False)
#                         rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
#                         lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
#                         cur_reward_sum[new_ids] = 0
#                         cur_episode_length[new_ids] = 0

#                 stop = time.time()
#                 collection_time = stop - start

#                 # Learning step
#                 start = stop
#                 self.alg.compute_returns(critic_obs)
            
#             mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred = self.alg.update()
#             stop = time.time()
#             learn_time = stop - start
#             if self.log_dir is not None:
#                 self.log(locals())
#             if it % self.save_interval == 0:
#                 self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))

#             wandb_dict = {
#                 "dt" : self.env.dt,
#                 "Train/mean_policy_pred": mean_policy_pred,
#                 "Train/mean_expert_pred": mean_expert_pred,
#             }
#             if it%10 == 0:
#                 self.log_wandb(wandb_dict, locals())
#             ep_infos.clear()

        
#         self.current_learning_iteration += num_learning_iterations
#         self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
#         self.save_wandb(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

#     def log(self, locs, width=80, pad=35):
#         self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
#         self.tot_time += locs['collection_time'] + locs['learn_time']
#         iteration_time = locs['collection_time'] + locs['learn_time']

#         ep_string = f''
#         if locs['ep_infos']:
#             for key in locs['ep_infos'][0]:
#                 infotensor = torch.tensor([], device=self.device)
#                 for ep_info in locs['ep_infos']:
#                     # handle scalar and zero dimensional tensor infos
#                     if not isinstance(ep_info[key], torch.Tensor):
#                         ep_info[key] = torch.Tensor([ep_info[key]])
#                     if len(ep_info[key].shape) == 0:
#                         ep_info[key] = ep_info[key].unsqueeze(0)
#                     infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
#                 value = torch.mean(infotensor)
#                 self.writer.add_scalar('Episode/' + key, value, locs['it'])
#                 ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
#         mean_std = self.alg.actor_critic.std.mean()
#         fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

#         self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
#         self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
#         self.writer.add_scalar('Loss/AMP', locs['mean_amp_loss'], locs['it'])
#         self.writer.add_scalar('Loss/AMP_grad', locs['mean_grad_pen_loss'], locs['it'])
#         self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
#         self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
#         self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
#         self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
#         self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
#         if len(locs['rewbuffer']) > 0:
#             self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
#             self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
#             self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
#             self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

#         str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

#         if len(locs['rewbuffer']) > 0:
#             log_string = (f"""{'#' * width}\n"""
#                           f"""{str.center(width, ' ')}\n\n"""
#                           f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
#                             'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
#                           f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
#                           f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
#                           f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
#                           f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
#                           f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
#                           f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
#                           f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
#                           f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
#                           f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
#                         #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
#                         #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
#         else:
#             log_string = (f"""{'#' * width}\n"""
#                           f"""{str.center(width, ' ')}\n\n"""
#                           f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
#                             'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
#                           f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
#                           f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
#                           f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
#                         #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
#                         #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

#         log_string += ep_string
#         log_string += (f"""{'-' * width}\n"""
#                        f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
#                        f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
#                        f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
#                        f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
#                                locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
#         print(log_string)

#     def save(self, path, infos=None):
#         torch.save({
#             'model_state_dict': self.alg.actor_critic.state_dict(),
#             'optimizer_state_dict': self.alg.optimizer.state_dict(),
#             'discriminator_state_dict': self.alg.discriminator.state_dict(),
#             'amp_normalizer': self.alg.amp_normalizer,
#             'iter': self.current_learning_iteration,
#             'infos': infos,
#             }, path)

#     def load(self, path, load_optimizer=True):
#         loaded_dict = torch.load(path)
#         self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
#         self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'])
#         self.alg.amp_normalizer = loaded_dict['amp_normalizer']
#         if load_optimizer:
#             self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
#         self.current_learning_iteration = loaded_dict['iter']
#         return loaded_dict['infos']

#     def get_inference_policy(self, device=None):
#         self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
#         if device is not None:
#             self.alg.actor_critic.to(device)
#         return self.alg.actor_critic.act_inference

#     def log_wandb(self, d, locs):
#         if self.LOG_WANDB:
#             wandb_dict = dict()
#             wandb_dict = {**wandb_dict, **d}
#             wandb_dict['n_updates'] = locs['it']
#             wandb_dict['Loss/value_function'] = locs['mean_value_loss']
#             wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
#             wandb_dict['Loss/AMP'] = locs['mean_amp_loss']
#             wandb_dict['Loss/AMP_grad'] = locs['mean_grad_pen_loss']
#             wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
#             if locs['ep_infos']:
#                 for key in locs['ep_infos'][0]:
#                     infotensor = torch.tensor([], device=self.device)
#                     for ep_info in locs['ep_infos']:
#                         # handle scalar and zero dimensional tensor infos
#                         if not isinstance(ep_info[key], torch.Tensor):
#                             ep_info[key] = torch.Tensor([ep_info[key]])
#                         if len(ep_info[key].shape) == 0:
#                             ep_info[key] = ep_info[key].unsqueeze(0)
#                         infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
#                     value = torch.mean(infotensor)
#                     wandb_dict['Train/Mean_episode_' + key] = value.item()
#             if len(locs['rewbuffer']) > 0:
#                 wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
#                 wandb_dict['Train/mean_episode_length_t'] = statistics.mean(locs['lenbuffer']) * wandb_dict['dt']
#                 wandb.log(wandb_dict)


#     def save_wandb(self, model_path):
#         if self.LOG_WANDB:
#             artifact = wandb.Artifact('model', type='model')
#             artifact.add_file(model_path)
#             wandb.run.log_artifact(artifact)
#             wandb.run.finish()