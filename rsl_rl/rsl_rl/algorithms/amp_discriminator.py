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
            amp_layers.append(nn.ReLU())
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
        return grad_pen

    def predict_amp_reward(
            self, state, next_state, task_reward, normalizer=None):
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)

            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            reward = self.amp_reward_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)
            if self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
            self.train()
        return reward.squeeze(dim=-1), d

    def _lerp_reward(self, disc_r, task_r):
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        return r
    
# Wasserstein critic
class AMPCritic(AMPDiscriminator):
    def __init__(
            self, input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp=0.0):
        super(AMPCritic, self).__init__(input_dim, amp_reward_coef, hidden_layer_sizes, device, task_reward_lerp)

    def compute_grad_pen(self,
                         expert_state,
                         expert_next_state,
                         policy_state,
                         policy_next_state,
                         lambda_=10):
        batch_size = expert_state.shape[0]
        blend = torch.rand((batch_size,))
        blend_state = self.blend(expert_state, policy_state, blend)
        blend_next_state = self.blend(expert_next_state, policy_next_state, blend)
        expert_data = torch.cat([blend_state, blend_next_state], dim=-1)
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad( # Computes the gradients of outputs w.r.t. inputs.
            outputs=disc, inputs=expert_data,
            grad_outputs=ones, create_graph=True,
            retain_graph=True, only_inputs=True)[0] # the index [0] indicates gradient w.r.t. the first input. For multiple inputs, use inputs=[input1, input2]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1.).pow(2).mean()
        return grad_pen

    def predict_amp_reward(
            self, state, next_state, task_reward, normalizer=None):
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)

            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            reward = self.amp_reward_coef * torch.exp(d)
            if self.task_reward_lerp > 0:
                reward = self._lerp_reward(reward, task_reward.unsqueeze(-1))
            self.train()
        return reward.squeeze(dim=-1), d
    

    def blend(self, frame0, frame1, blend):
        root_pos0, root_pos1 = AMPLoader.get_root_pos_batch(frame0), AMPLoader.get_root_pos_batch(frame1)
        root_rot0, root_rot1 = AMPLoader.get_root_rot_batch(frame0), AMPLoader.get_root_rot_batch(frame1)
        linear_vel_0, linear_vel_1 = AMPLoader.get_linear_vel_batch(frame0), AMPLoader.get_linear_vel_batch(frame1)
        angular_vel_0, angular_vel_1 = AMPLoader.get_angular_vel_batch(frame0), AMPLoader.get_angular_vel_batch(frame1)
        joints0, joints1 = AMPLoader.get_joint_pose_batch(frame0), AMPLoader.get_joint_pose_batch(frame1)
        joint_vel_0, joint_vel_1 = AMPLoader.get_joint_vel_batch(frame0), AMPLoader.get_joint_vel_batch(frame1)
        foot_pos0, foot_pos1 = AMPLoader.get_foot_pos_batch(frame0), AMPLoader.get_foot_pos_batch(frame1)
        Lfoot_rot0, Lfoot_rot1 = AMPLoader.get_foot_rot_batch(frame0)[0:AMPLoader.ROT_SIZE], AMPLoader.get_foot_rot_batch(frame1)[0:AMPLoader.ROT_SIZE]
        Rfoot_rot0, Rfoot_rot1 = AMPLoader.get_foot_rot_batch(frame0)[AMPLoader.ROT_SIZE:], AMPLoader.get_foot_rot_batch(frame1)[AMPLoader.ROT_SIZE:]
        
        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        blend_root_rot = self.slerp(root_rot0, root_rot1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)
        blend_foot_pos = self.slerp(foot_pos0, foot_pos1, blend)
        blend_Lfoot_rot = self.quaternion_slerp(
            Lfoot_rot0.cpu().numpy(), Lfoot_rot1.cpu().numpy(), blend)
        blend_Rfoot_rot = self.quaternion_slerp(
            Rfoot_rot0.cpu().numpy(), Rfoot_rot1.cpu().numpy(), blend)
        blend_Lfoot_rot[:, -1] = blend_Lfoot_rot[:, -1].abs()
        blend_Rfoot_rot[:, -1] = blend_Rfoot_rot[:, -1].abs()

        ret = torch.cat([
            blend_root_pos, blend_root_rot, blend_linear_vel, blend_angular_vel, blend_joints, blend_joints_vel,
            blend_foot_pos, blend_Lfoot_rot, blend_Rfoot_rot
            ], dim=-1)
        assert ret.shape == frame0.shape, f"Blended shape is wrong. {ret.shape}"
        return ret        
    
    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1
    
    def quaternion_slerp(self, q0, q1, fraction):
        return utils.quaternion_slerp(q0, q1, fraction)
