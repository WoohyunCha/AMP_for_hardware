# import torch
# import torch.nn as nn
# import torch.utils.data
# from torch import autograd

# from rsl_rl.utils import utils


# class AMPDiscriminator(nn.Module):
#     def __init__(
#             self, input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp=0.0):
#         super(AMPDiscriminator, self).__init__()

#         self.device = device
#         self.input_dim = input_dim

#         self.amp_reward_coef = amp_reward_coef
#         amp_layers = []
#         curr_in_dim = input_dim
#         for hidden_dim in hidden_layer_sizes:
#             amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
#             amp_layers.append(nn.ReLU())
#             curr_in_dim = hidden_dim
#         self.trunk = nn.Sequential(*amp_layers).to(device)
#         self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

#         self.trunk.train()
#         self.amp_linear.train()

#         self.task_reward_lerp = task_reward_lerp

#     def forward(self, x):
#         h = self.trunk(x)
#         d = self.amp_linear(h)
#         return d

#     def compute_grad_pen(self,
#                          expert_state,
#                          expert_next_state,
#                          lambda_=10):
#         expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
#         expert_data.requires_grad = True

#         disc = self.amp_linear(self.trunk(expert_data))
#         ones = torch.ones(disc.size(), device=disc.device)
#         grad = autograd.grad( # Computes the gradients of outputs w.r.t. inputs.
#             outputs=disc, inputs=expert_data,
#             grad_outputs=ones, create_graph=True,
#             retain_graph=True, only_inputs=True)[0] # the index [0] indicates gradient w.r.t. the first input. For multiple inputs, use inputs=[input1, input2]

#         # Enforce that the grad norm approaches 0.
#         grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
#         return grad_pen

#     def predict_amp_reward(
#             self, state, next_state, task_reward, normalizer=None):
#         with torch.no_grad():
#             self.eval()
#             if normalizer is not None:
#                 state = normalizer.normalize_torch(state, self.device)
#                 next_state = normalizer.normalize_torch(next_state, self.device)

#             d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
#             reward = self.amp_reward_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)
#             if self.task_reward_lerp > 0:
#                 reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
#             self.train()
#         return reward.squeeze(dim=-1), d

#     def _lerp_reward(self, disc_r, task_r):
#         r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
#         return r

import torch
import torch.nn as nn
import torch.utils.data
from torch import autograd

from rsl_rl.utils import utils
from rsl_rl.datasets.motion_loader import AMPLoader

class AMPDiscriminator(nn.Module):
    def __init__(
            self, input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp=0.0):
        super(AMPDiscriminator, self).__init__()

        self.device = device
        self.input_dim = input_dim

        self.amp_reward_coef = amp_reward_coef
        amp_layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU()) # Relu is more stable?
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.amp_linear.train()

        self.task_reward_lerp = task_reward_lerp

    def forward(self, x):
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self,
                         expert_state,
                         expert_next_state,
                         lambda_=10):
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad( # Computes the gradients of outputs w.r.t. inputs.
            outputs=disc, inputs=expert_data,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0] # the index [0] indicates gradient w.r.t. the first input. For multiple inputs, use inputs=[input1, input2]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        # grad_pen = lambda_ * torch.sum(torch.square(grad), dim=-1).mean()
        return grad_pen

    def predict_amp_reward(
            self, state, next_state, task_reward, normalizer=None):
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                if state.shape == next_state.shape:
                    next_state = normalizer.normalize_torch(next_state, self.device)
                else:
                    next_state[:, :state.shape[-1]] = normalizer.normalize_torch(next_state[:, :state.shape[-1]], self.device)
                    
            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            reward = self.amp_reward_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)
            # print(f"amp reward : {reward}")
            # print(f"task reward : {task_reward}")
            amp_reward = reward
            if self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
            self.train()
        return reward.squeeze(dim=-1), amp_reward.squeeze(dim=-1), d

    def _lerp_reward(self, disc_r, task_r):
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r
    
# Wasserstein critic
class AMPCritic(AMPDiscriminator):
    def __init__(
            self, input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp=0.0):
        super(AMPCritic, self).__init__(input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp)
        print("AMP WGAN CRITIC INITIALIZED")

    def predict_amp_reward(
            self, state, next_state, task_reward, normalizer=None):
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                if state.shape == next_state.shape:
                    next_state = normalizer.normalize_torch(next_state, self.device)
                else:
                    next_state[:, :state.shape[-1]] = normalizer.normalize_torch(next_state[:, :state.shape[-1]], self.device)
                # next_state = normalizer.normalize_torch(next_state, self.device)


            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            reward = self.amp_reward_coef * .5 * torch.exp(d)
            amp_reward = reward
            if self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
            self.train()
        return reward.squeeze(dim=-1), amp_reward.squeeze(dim=-1), d 
    
    def compute_grad_pen_interpolate(
            self,
            expert_state,
            expert_next_state,
            policy_state,
            policy_next_state,
            lambda_=10
    ):
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        # expert_data.requires_grad = True
        policy_data = torch.cat([policy_state, policy_next_state], dim=-1)
        interpolate_data = slerp(expert_data, policy_data, torch.rand((expert_data.shape[0], 1), device=self.device))
        interpolate_data.requires_grad = True

        disc = self.amp_linear(self.trunk(interpolate_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad( # Computes the gradients of outputs w.r.t. inputs.
            outputs=disc, inputs=interpolate_data,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0] # the index [0] indicates gradient w.r.t. the first input. For multiple inputs, use inputs=[input1, input2]

        # Enforce that the grad norm approaches 0.
        # grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        grad_pen = lambda_ * (torch.clamp(grad.norm(2, dim=1) - 1., min=0.)).pow(2).mean()
        return grad_pen



    # def compute_grad_pen(self,
    #                      expert_state,
    #                      expert_next_state,
    #                      lambda_=10):
        # expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        # expert_data.requires_grad = True

        # disc = self.amp_linear(self.trunk(expert_data))
        # ones = torch.ones(disc.size(), device=disc.device)
        # grad = autograd.grad( # Computes the gradients of outputs w.r.t. inputs.
        #     outputs=disc, inputs=expert_data,
        #     grad_outputs=ones, create_graph=True,
        #     retain_graph=True, only_inputs=True)[0] # the index [0] indicates gradient w.r.t. the first input. For multiple inputs, use inputs=[input1, input2]

        # # Enforce that the grad norm approaches 0.
        # # grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        # grad_pen = lambda_ * torch.square(torch.clamp(grad.norm(2, dim=1) - 1., min=0.)).mean()
        # return grad_pen

def slerp(a, b, blend):
    return (1-blend)*a + blend*b