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

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage, RolloutStorage_history
from rsl_rl.storage.replay_buffer import ReplayBuffer
from rsl_rl.algorithms.amp_discriminator import AMPCritic
from rsl_rl.modules.encoder import CNNEncoder

class AMPPPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 discriminator,
                 amp_data,
                 amp_normalizer,
                 disc_grad_pen,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 amp_replay_buffer_size=100000,
                 min_std=None,
                 disc_coef=5.,
                 bounds_loss_coef=10.,
                 encoder_dim = None,
                 encoder_history_steps = 50
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.min_std = min_std

        # Discriminator components
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_transition = RolloutStorage.Transition()
        self.amp_storage = ReplayBuffer(
            discriminator.input_dim // 2, amp_replay_buffer_size, device)
        self.amp_data = amp_data # AMPLoader
        self.amp_normalizer = amp_normalizer

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later

        self.encoder_dim = encoder_dim if encoder_dim is not None else 0 # encoder
        self.encoder_history_steps = encoder_history_steps

        # Optimizer for policy and discriminator.
        params = [
            {'params': self.actor_critic.parameters(), 'name': 'actor_critic'},
            {'params': self.discriminator.trunk.parameters(),
             'weight_decay': 10e-4, 'name': 'amp_trunk'},
            {'params': self.discriminator.amp_linear.parameters(),
             'weight_decay': 10e-2, 'name': 'amp_head'}]

        self.optimizer = optim.Adam(params, lr=learning_rate)
        if self.encoder_dim: # encoder
            self.transition = RolloutStorage_history.Transition()
        else:
            self.transition = RolloutStorage.Transition()      


        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.disc_coef = disc_coef
        self.disc_grad_pen = disc_grad_pen
        self.bounds_loss_coef = bounds_loss_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, history_shape = None):
        if self.encoder_dim: # encoder
            self.storage = RolloutStorage_history(
                num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, history_shape, self.device)
        else:          
            self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)


    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, amp_obs, long_history = None):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        aug_obs, aug_critic_obs = obs.detach(), critic_obs.detach()
        #encoder
        if long_history is not None: # received history vector. 
            # latent_vector= self.encoder(long_history).detach()
            # assert latent_vector.shape == (aug_obs.shape[0], self.encoder_dim), f"laten vector shape : {latent_vector.shape}"
            self.transition.actions = self.actor_critic.act(aug_obs, long_history.detach()).detach()
            self.transition.values = self.actor_critic.evaluate(aug_obs, long_history.detach()).detach()
            self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
            self.transition.action_mean = self.actor_critic.action_mean.detach()
            self.transition.action_sigma = self.actor_critic.action_std.detach()
            # need to record obs and critic_obs before env.step()
            self.transition.observations = obs
            self.transition.critic_observations = critic_obs
            self.transition.history  = long_history
            self.amp_transition.observations = amp_obs
            return self.transition.actions
        self.transition.actions = self.actor_critic.act(aug_obs).detach()
        self.transition.values = self.actor_critic.evaluate(aug_critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.amp_transition.observations = amp_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos, amp_obs):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        not_done_idxs = (dones == False).nonzero().squeeze()
        self.amp_storage.insert(
            self.amp_transition.observations, amp_obs)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs, long_history):
        if self.encoder_dim:
            aug_last_critic_obs = last_critic_obs.detach()
            last_values = self.actor_critic.evaluate(aug_last_critic_obs, long_history.detach()).detach()
            self.storage.compute_returns(last_values, self.gamma, self.lam)
        else:
            aug_last_critic_obs = last_critic_obs.detach()
            last_values = self.actor_critic.evaluate(aug_last_critic_obs).detach()
            self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
                self.num_mini_batches)
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
                self.num_mini_batches)
        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):

                obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                    old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch = sample
                aug_obs_batch = obs_batch.detach()
                self.actor_critic.act(aug_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                aug_critic_obs_batch = critic_obs_batch.detach()
                value_batch = self.actor_critic.evaluate(aug_critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-6, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                # ratio = torch.exp(-actions_log_prob_batch + torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
                
                # bound loss
                soft_bound = 1.0
                mu_loss_high = torch.maximum(mu_batch - soft_bound,
                                             torch.tensor(0, device=self.device)) ** 2 
                mu_loss_low = torch.minimum(mu_batch + soft_bound, torch.tensor(0, device=self.device)) ** 2
                b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                
                # Discriminator loss.
                policy_state, policy_next_state = sample_amp_policy
                expert_state, expert_next_state = sample_amp_expert
                if self.amp_normalizer is not None:
                    with torch.no_grad():
                        policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                        policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                        expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                        expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
                policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
                expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
                if isinstance(self.discriminator, AMPCritic):
                    # WGAN
                    boundary = 0.5 # 0.1 ~ 0.5 is the proper range of selection
                    expert_loss = -torch.nn.Tanh()(
                        boundary*expert_d
                    ).mean()
                    policy_loss = torch.nn.Tanh()(
                        boundary*policy_d
                    ).mean()
                    amp_loss = (expert_loss + policy_loss) 
                    grad_pen_loss = self.discriminator.compute_grad_pen(
                        *sample_amp_expert, lambda_=self.disc_grad_pen)
                else: 
                    # LSGAN
                    expert_loss = torch.nn.MSELoss()(
                        expert_d, torch.ones(expert_d.size(), device=self.device))
                    policy_loss = torch.nn.MSELoss()(
                        policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
                    amp_loss = (expert_loss + policy_loss) *0.5
                    grad_pen_loss = self.discriminator.compute_grad_pen( 
                        *sample_amp_expert, lambda_=self.disc_grad_pen) 
                # logit reg
                logit_weights = torch.flatten(self.discriminator.amp_linear.weight)
                disc_logit_loss = 0.05*torch.sum(torch.square(logit_weights))
                # # weight decay
                # weights = []
                # for m in self.discriminator.trunk.modules():
                #     if isinstance(m, nn.Linear):
                #         weights.append(torch.flatten(m.weight))
                # weights.append(logit_weights)
                # disc_weights = torch.cat(weights, dim=-1)
                # disc_weight_decay = 0.0001*torch.sum(torch.square(disc_weights))

                # Compute total loss.
                loss = (
                    surrogate_loss +
                    self.value_loss_coef * value_loss +
                    self.bounds_loss_coef * b_loss.mean() -
                    self.entropy_coef * entropy_batch.mean() +
                    self.disc_coef*(amp_loss + grad_pen_loss + disc_logit_loss)
                    )
                
                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()



                if not self.actor_critic.fixed_std and self.min_std is not None:
                    self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)

                if self.amp_normalizer is not None:
                    self.amp_normalizer.update(policy_state.cpu().numpy())
                    self.amp_normalizer.update(expert_state.cpu().numpy())

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_amp_loss += amp_loss.item()
                mean_grad_pen_loss += grad_pen_loss.item()
                mean_policy_pred += policy_d.mean().item()
                mean_expert_pred += expert_d.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred

    def update_latent(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        if self.actor_critic.is_recurrent:
            raise NotImplementedError
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
                self.num_mini_batches)
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
                self.num_mini_batches)
        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):

                obs_batch, history_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                    old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch = sample
                aug_obs_batch = obs_batch.detach()
                long_history_batch = history_batch.detach()
                self.actor_critic.act(aug_obs_batch, long_history_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                aug_critic_obs_batch = critic_obs_batch.detach()
                value_batch = self.actor_critic.evaluate(aug_critic_obs_batch, long_history_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)
                        
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-6, self.learning_rate / 1.5)

                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        for param_group in self.optimizer.param_groups:
                            # if param_group['name'] == 'encoder':
                            #     lr = param_group['lr']
                            #     if kl_mean > self.desired_kl * 2.0:
                            #         param_group['lr'] = max(1e-6, lr / 1.5)

                            #     elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            #         param_group['lr'] = min(1e-2, lr * 1.5)                               
                            # else:
                            param_group['lr'] = self.learning_rate
                            



                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                # ratio = torch.exp(-actions_log_prob_batch + torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
                
                # bound loss
                soft_bound = 1.0
                mu_loss_high = torch.maximum(mu_batch - soft_bound,
                                             torch.tensor(0, device=self.device)) ** 2 
                mu_loss_low = torch.minimum(mu_batch + soft_bound, torch.tensor(0, device=self.device)) ** 2
                b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                
                # Discriminator loss.
                policy_state, policy_next_state = sample_amp_policy
                expert_state, expert_next_state = sample_amp_expert
                if self.amp_normalizer is not None:
                    with torch.no_grad():
                        policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                        policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                        expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                        expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
                policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
                expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
                if isinstance(self.discriminator, AMPCritic):
                    # WGAN
                    boundary = 0.5 # 0.1 ~ 0.5 is the proper range of selection
                    expert_loss = -torch.nn.Tanh()(
                        boundary*expert_d
                    ).mean()
                    policy_loss = torch.nn.Tanh()(
                        boundary*policy_d
                    ).mean()
                    amp_loss = (expert_loss + policy_loss) *0.5
                    grad_pen_loss = self.discriminator.compute_grad_pen(
                        *sample_amp_expert, lambda_=self.disc_grad_pen)
                else: 
                    # LSGAN
                    expert_loss = torch.nn.MSELoss()(
                        expert_d, torch.ones(expert_d.size(), device=self.device))
                    policy_loss = torch.nn.MSELoss()(
                        policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
                    amp_loss = (expert_loss + policy_loss) *0.5
                    grad_pen_loss = self.discriminator.compute_grad_pen( 
                        *sample_amp_expert, lambda_=self.disc_grad_pen) 
                # logit reg
                logit_weights = torch.flatten(self.discriminator.amp_linear.weight)
                disc_logit_loss = 0.05*torch.sum(torch.square(logit_weights))
                # # weight decay
                # weights = []
                # for m in self.discriminator.trunk.modules():
                #     if isinstance(m, nn.Linear):
                #         weights.append(torch.flatten(m.weight))
                # weights.append(logit_weights)
                # disc_weights = torch.cat(weights, dim=-1)
                # disc_weight_decay = 0.0001*torch.sum(torch.square(disc_weights))

                # Compute total loss.
                loss = (
                    surrogate_loss +
                    self.value_loss_coef * value_loss +
                    self.bounds_loss_coef * b_loss.mean() -
                    self.entropy_coef * entropy_batch.mean() +
                    self.disc_coef*(amp_loss + grad_pen_loss + disc_logit_loss)
                    )
                
                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if not self.actor_critic.fixed_std and self.min_std is not None:
                    self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)

                if self.amp_normalizer is not None:
                    self.amp_normalizer.update(policy_state.cpu().numpy())
                    self.amp_normalizer.update(expert_state.cpu().numpy())

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_amp_loss += amp_loss.item()
                mean_grad_pen_loss += grad_pen_loss.item()
                mean_policy_pred += policy_d.mean().item()
                mean_expert_pred += expert_d.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred


    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.0
            mu_loss_high = torch.maximum(mu - soft_bound, torch.tensor(0, device=self.ppo_device))**2
            mu_loss_low = torch.minimum(mu + soft_bound, torch.tensor(0, device=self.ppo_device))**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss
    
class AMPPPOSym(AMPPPO):

    def __init__(self, actor_critic, discriminator, amp_data, amp_normalizer, disc_grad_pen, 
                 num_learning_epochs=1, num_mini_batches=1, clip_param=0.2, gamma=0.998, lam=0.95, value_loss_coef=1, entropy_coef=0, 
                 learning_rate=0.001, max_grad_norm=1, use_clipped_value_loss=True, schedule="fixed", desired_kl=0.01, device='cpu', amp_replay_buffer_size=100000, 
                 min_std=None, disc_coef=5, bounds_loss_coef=10, encoder_dim=None, encoder_history_steps=50,
                include_history_steps = 1,
                mirror = {},
                mirror_weight = 4., 
                num_actuation = None, 
                mirror_neg = {}, 
                cartesian_angular_mirror=[], 
                cartesian_linear_mirror=[], 
                cartesian_command_mirror=[], 
                switch_mirror = [], 
                phase_mirror = [], 
                no_mirror = []
                ):
        super().__init__(actor_critic, discriminator, amp_data, amp_normalizer, disc_grad_pen, num_learning_epochs, num_mini_batches, clip_param, gamma, 
                         lam, value_loss_coef, entropy_coef, learning_rate, max_grad_norm, use_clipped_value_loss, schedule, desired_kl, device, 
                         amp_replay_buffer_size, min_std, disc_coef, bounds_loss_coef, encoder_dim, encoder_history_steps)
        print("SYM AMP PPO LOADED!!")
        self.include_history_steps = include_history_steps
        self.encoder_dim = encoder_dim if encoder_dim is not None else 0
        self.mirror_dict = mirror
        self.mirror_neg_dict = mirror_neg
        self.cartesian_angular_mirror = cartesian_angular_mirror
        self.cartesian_linear_mirror = cartesian_linear_mirror
        self.cartesian_command_mirror = cartesian_command_mirror
        self.switch_mirror = switch_mirror
        self.phase_mirror = phase_mirror
        self.no_mirror = no_mirror
        self.mirror_weight = mirror_weight
        self.mirror_init = True
        self.num_actuation = num_actuation

    def get_mirror_loss(self, obs_batch, actions_batch, long_history_batch):
        num_obvs = int(obs_batch.shape[1]/ self.include_history_steps) # length of each observation : 48
        num_acts = actions_batch.shape[1]
        if self.mirror_init:
            print("MIRROR TEST")
            minibatchsize = obs_batch.shape[0]
            cartesian_mirror_count = 0
            no_mirror_count = 0
            switch_mirror_count = 0
            phase_mirror_count = 0
            self.mirror_obs = torch.eye(num_obvs).to(device=self.device)
            self.mirror_act = torch.eye(num_acts).to(device=self.device)
            # Joint space mirrors

            for _, (i,j) in self.mirror_dict.items():
                self.mirror_act[i, i] = 0
                self.mirror_act[j, j] = 0
                self.mirror_act[i, j] = 1
                self.mirror_act[j, i] = 1
            for _, (i, j) in self.mirror_neg_dict.items():
                self.mirror_act[i, i] = 0
                self.mirror_act[j, j] = 0
                self.mirror_act[i, j] = -1
                self.mirror_act[j, i] = -1
                # Cartesian space mirrors

            for (start, atend) in self.cartesian_angular_mirror:
                # cartesian mirrors from range(start, atend)
                if (atend-start)%3==0:
                    for i in range(int((atend-start)/3)):
                        self.mirror_obs[start + 3*i, start + 3*i] *= -1
                        self.mirror_obs[start+2 + 3*i, start+2 + 3*i] *= -1
                        cartesian_mirror_count += 3
                else:
                    raise ValueError("SOMETHING WRONG IN CARTESIAN SPACE MIRRORS!!(angular)")
            for (start, atend) in self.cartesian_linear_mirror:
                if (atend-start)%3==0:
                    for i in range(int((atend-start)/3)):
                        self.mirror_obs[start+1+ 3*i, start+1+ 3*i] *= -1
                        cartesian_mirror_count += 3
                else:
                    raise ValueError("SOMETHING WRONG IN CARTESIAN SPACE MIRRORS!!(linear)")                        
                    
            for (start, atend) in self.cartesian_command_mirror:
                if (atend-start)%3==0:
                    for i in range(int((atend-start)/3)):
                        self.mirror_obs[start+1+ 3*i, start+1+ 3*i] *= -1
                        self.mirror_obs[start+2+ 3*i, start+2+ 3*i] *= -1
                        cartesian_mirror_count += 3  
                else:
                    raise ValueError("SOMETHING WRONG IN CARTESIAN SPACE MIRRORS!!(command)")
            for (start, atend) in self.switch_mirror:
                if (atend-start)%2==0:
                    for i in range(int((atend-start)/2)):
                        self.mirror_obs[start+2*i, start+2*i] *= 0
                        self.mirror_obs[start+2*i+1, start+2*i+1] *= 0
                        self.mirror_obs[start+2*i+1, start+2*i] = 1
                        self.mirror_obs[start+2*i, start+2*i+1] = 1
                        
                        switch_mirror_count += 2
                else:
                    raise ValueError("SOMETHING WRONG IN SWITCH MIRRORS!!")          
            for (start, atend) in self.phase_mirror:
                assert atend-start == 2, "Improper phase representation. A single (cos, sin) pair"
                for i in range(int(atend-start)):
                    self.mirror_obs[:, start+i, start+i] = -1
                    phase_mirror_count += 1
            for (start, atend) in self.no_mirror:
                for _ in range(start, atend):
                    no_mirror_count += 1
            # Joint space mirroring
            assert ((num_obvs - cartesian_mirror_count - switch_mirror_count - no_mirror_count - phase_mirror_count) % num_acts) == 0, f"something wrong in mirror total1 {(num_obvs - cartesian_mirror_count - switch_mirror_count - no_mirror_count - phase_mirror_count)} cannot be divided by {num_acts}"
            for i in range(int((num_obvs - cartesian_mirror_count - switch_mirror_count - phase_mirror_count - no_mirror_count) / num_acts)):
                self.mirror_obs[cartesian_mirror_count + i*num_acts:cartesian_mirror_count + (i+1)*num_acts, cartesian_mirror_count + i*num_acts:cartesian_mirror_count + (i+1)*num_acts] = self.mirror_act

            print("------ABOUT MIRROR------")
            print("Total number of elements of cartesian space mirroring : ", cartesian_mirror_count)
            print("Total number of elements of no mirroring : ", no_mirror_count)
            print("Total number of elements of joint space mirroring : ", num_obvs - cartesian_mirror_count - no_mirror_count)                    
            self.mirror_init = False
        
        obs_batch, actions_batch = obs_batch.detach(), actions_batch.detach()
        if long_history_batch is not None:
            long_history_batch = long_history_batch.detach()
            mirror_history_obs_batch = self.mirror_obs @ long_history_batch

        mirror_obs_batch = torch.flatten(self.mirror_obs @ obs_batch.view(-1, num_obvs, self.include_history_steps), start_dim=1)
        if long_history_batch is not None:
            mirrored_actions = self.actor_critic.act_inference(mirror_obs_batch, mirror_history_obs_batch).unsqueeze(2)
        else:
            mirrored_actions = self.actor_critic.act_inference(mirror_obs_batch).unsqueeze(2)
        mirrored_actions : torch.Tensor = self.mirror_act@mirrored_actions
        assert mirrored_actions.requires_grad, f"gradient lost in mirror!"
        if long_history_batch  is not None:
            return nn.MSELoss()(self.actor_critic.act_inference(obs_batch, long_history_batch), mirrored_actions.squeeze())
        else:
            return nn.MSELoss()(self.actor_critic.act_inference(obs_batch), mirrored_actions.squeeze())

    def update_latent(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        mean_mirror_loss = 0
        if self.actor_critic.is_recurrent:
            raise NotImplementedError
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
                self.num_mini_batches)
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
                self.num_mini_batches)
        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):

                obs_batch, history_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                    old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch = sample
                aug_obs_batch = obs_batch.detach()
                long_history_batch = history_batch.detach()
                self.actor_critic.act(aug_obs_batch, long_history_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                aug_critic_obs_batch = critic_obs_batch.detach()
                value_batch = self.actor_critic.evaluate(aug_critic_obs_batch, long_history_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)
                        
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-6, self.learning_rate / 1.5)

                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        for param_group in self.optimizer.param_groups:
                            # if param_group['name'] == 'encoder':
                            #     lr = param_group['lr']
                            #     if kl_mean > self.desired_kl * 2.0:
                            #         param_group['lr'] = max(1e-6, lr / 1.5)

                            #     elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            #         param_group['lr'] = min(1e-2, lr * 1.5)                               
                            # else:
                            param_group['lr'] = self.learning_rate
                            



                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                # ratio = torch.exp(-actions_log_prob_batch + torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
                
                # bound loss
                soft_bound = 1.0
                mu_loss_high = torch.maximum(mu_batch - soft_bound,
                                             torch.tensor(0, device=self.device)) ** 2 
                mu_loss_low = torch.minimum(mu_batch + soft_bound, torch.tensor(0, device=self.device)) ** 2
                b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                
                # Discriminator loss.
                policy_state, policy_next_state = sample_amp_policy
                expert_state, expert_next_state = sample_amp_expert
                if self.amp_normalizer is not None:
                    with torch.no_grad():
                        policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                        policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                        expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                        expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
                policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
                expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
                if isinstance(self.discriminator, AMPCritic):
                    # WGAN
                    boundary = 0.5 # 0.1 ~ 0.5 is the proper range of selection
                    expert_loss = -torch.nn.Tanh()(
                        boundary*expert_d
                    ).mean()
                    policy_loss = torch.nn.Tanh()(
                        boundary*policy_d
                    ).mean()
                    amp_loss = (expert_loss + policy_loss) *0.5
                    grad_pen_loss = self.discriminator.compute_grad_pen(
                        *sample_amp_expert, lambda_=self.disc_grad_pen)
                else: 
                    # LSGAN
                    expert_loss = torch.nn.MSELoss()(
                        expert_d, torch.ones(expert_d.size(), device=self.device))
                    policy_loss = torch.nn.MSELoss()(
                        policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
                    amp_loss = (expert_loss + policy_loss) *0.5
                    grad_pen_loss = self.discriminator.compute_grad_pen( 
                        *sample_amp_expert, lambda_=self.disc_grad_pen) 
                # logit reg
                logit_weights = torch.flatten(self.discriminator.amp_linear.weight)
                disc_logit_loss = 0.05*torch.sum(torch.square(logit_weights))
                # # weight decay
                # weights = []
                # for m in self.discriminator.trunk.modules():
                #     if isinstance(m, nn.Linear):
                #         weights.append(torch.flatten(m.weight))
                # weights.append(logit_weights)
                # disc_weights = torch.cat(weights, dim=-1)
                # disc_weight_decay = 0.0001*torch.sum(torch.square(disc_weights))

                # Compute total loss.
                mirror_loss = 0.
                if self.mirror_weight > 0.:
                    mirror_loss = self.get_mirror_loss(obs_batch, actions_batch, long_history_batch)
                loss = (
                    surrogate_loss +
                    self.value_loss_coef * value_loss +
                    self.bounds_loss_coef * b_loss.mean() -
                    self.entropy_coef * entropy_batch.mean() +
                    self.disc_coef*(amp_loss + grad_pen_loss + disc_logit_loss)
                    + self.mirror_weight*mirror_loss
                    )
                
                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if not self.actor_critic.fixed_std and self.min_std is not None:
                    self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)

                if self.amp_normalizer is not None:
                    self.amp_normalizer.update(policy_state.cpu().numpy())
                    self.amp_normalizer.update(expert_state.cpu().numpy())

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_amp_loss += amp_loss.item()
                mean_grad_pen_loss += grad_pen_loss.item()
                mean_policy_pred += policy_d.mean().item()
                mean_expert_pred += expert_d.mean().item()
                mean_mirror_loss += mirror_loss

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        mean_mirror_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred, mean_mirror_loss

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        mean_mirror_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
                self.num_mini_batches)
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
                self.num_mini_batches)
        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):

                obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                    old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch = sample
                aug_obs_batch = obs_batch.detach()
                self.actor_critic.act(aug_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                aug_critic_obs_batch = critic_obs_batch.detach()
                value_batch = self.actor_critic.evaluate(aug_critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-6, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                # ratio = torch.exp(-actions_log_prob_batch + torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
                
                # bound loss
                soft_bound = 1.0
                mu_loss_high = torch.maximum(mu_batch - soft_bound,
                                             torch.tensor(0, device=self.device)) ** 2 
                mu_loss_low = torch.minimum(mu_batch + soft_bound, torch.tensor(0, device=self.device)) ** 2
                b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                
                # Discriminator loss.
                policy_state, policy_next_state = sample_amp_policy
                expert_state, expert_next_state = sample_amp_expert
                if self.amp_normalizer is not None:
                    with torch.no_grad():
                        policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                        policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                        expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                        expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
                policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
                expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
                if isinstance(self.discriminator, AMPCritic):
                    # WGAN
                    boundary = 0.5 # 0.1 ~ 0.5 is the proper range of selection
                    expert_loss = -torch.nn.Tanh()(
                        boundary*expert_d
                    ).mean()
                    policy_loss = torch.nn.Tanh()(
                        boundary*policy_d
                    ).mean()
                    amp_loss = (expert_loss + policy_loss) *0.5
                    grad_pen_loss = self.discriminator.compute_grad_pen(
                        *sample_amp_expert, lambda_=self.disc_grad_pen)
                else: 
                    # LSGAN
                    expert_loss = torch.nn.MSELoss()(
                        expert_d, torch.ones(expert_d.size(), device=self.device))
                    policy_loss = torch.nn.MSELoss()(
                        policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
                    amp_loss = (expert_loss + policy_loss) *0.5
                    grad_pen_loss = self.discriminator.compute_grad_pen( 
                        *sample_amp_expert, lambda_=self.disc_grad_pen) 
                # logit reg
                logit_weights = torch.flatten(self.discriminator.amp_linear.weight)
                disc_logit_loss = 0.05*torch.sum(torch.square(logit_weights))
                # # weight decay
                # weights = []
                # for m in self.discriminator.trunk.modules():
                #     if isinstance(m, nn.Linear):
                #         weights.append(torch.flatten(m.weight))
                # weights.append(logit_weights)
                # disc_weights = torch.cat(weights, dim=-1)
                # disc_weight_decay = 0.0001*torch.sum(torch.square(disc_weights))

                # Compute total loss.
                mirror_loss = self.get_mirror_loss(aug_obs_batch, actions_batch, None)
                loss = (
                    surrogate_loss +
                    self.value_loss_coef * value_loss +
                    self.bounds_loss_coef * b_loss.mean() -
                    self.entropy_coef * entropy_batch.mean() +
                    self.disc_coef*(amp_loss + grad_pen_loss + disc_logit_loss)
                    + self.mirror_weight * mirror_loss
                    )
                
                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()



                if not self.actor_critic.fixed_std and self.min_std is not None:
                    self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)

                if self.amp_normalizer is not None:
                    self.amp_normalizer.update(policy_state.cpu().numpy())
                    self.amp_normalizer.update(expert_state.cpu().numpy())

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_amp_loss += amp_loss.item()
                mean_grad_pen_loss += grad_pen_loss.item()
                mean_policy_pred += policy_d.mean().item()
                mean_expert_pred += expert_d.mean().item()
                mean_mirror_loss += mirror_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        mean_mirror_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred, mean_mirror_loss