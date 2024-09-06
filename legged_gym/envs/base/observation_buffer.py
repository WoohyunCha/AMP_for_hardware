import torch

BUFFER_SIZE_FACTOR = 10

class ObservationBuffer:
    def __init__(self, num_envs, num_obs, include_history_steps, skips, device):

        self.num_envs = num_envs
        self.num_obs = num_obs
        self.include_history_steps = include_history_steps
        self.device = device
        self.skips = skips # The number of skips between observations
        if self.skips is None:
            self.skips = 1
        # self.num_obs_total = num_obs * ((include_history_steps-1) * self.skips + 1)
        # self.num_obs_total = num_obs * (include_history_steps * self.skips)
        self.num_obs_total = num_obs * (include_history_steps * BUFFER_SIZE_FACTOR)

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs_total, device=self.device, dtype=torch.float)

    def reset(self, reset_idxs, new_obs):
        # self.obs_buf[reset_idxs] = new_obs.repeat(1, ((self.include_history_steps-1) * self.skips + 1))
        # self.obs_buf[reset_idxs] = new_obs.repeat(1, (self.include_history_steps * self.skips))
        self.obs_buf[reset_idxs] = new_obs.repeat(1, (self.include_history_steps * BUFFER_SIZE_FACTOR))

    def insert(self, new_obs):
        # # Shift observations back.
        # self.obs_buf[:, : self.num_obs * (self.include_history_steps - 1)] = self.obs_buf[:,self.num_obs : self.num_obs * self.include_history_steps]
        # # Add new observation.
        # self.obs_buf[:, -self.num_obs:] = new_obs
        # return
        # Shift observations back.
        self.obs_buf[:, : -self.num_obs] = self.obs_buf[:,self.num_obs:]
        # Add new observation.
        self.obs_buf[:, -self.num_obs:] = new_obs
        return


    def get_obs_vec(self, obs_ids, delay: torch.Tensor):
        """Gets history of observations indexed by obs_ids.
        
        Arguments:
            obs_ids: An array of integers with which to index the desired
                observations, where 0 is the latest observation and
                include_history_steps - 1 is the oldest observation.
        """

        # obs = []
        # for obs_id in reversed(sorted(obs_ids)):
        #     slice_idx = self.include_history_steps - obs_id - 1
        #     obs.append(self.obs_buf[:, self.skips * slice_idx * self.num_obs : (self.skips * slice_idx + 1) * self.num_obs])
        # return torch.cat(obs, dim=-1)

        # obs = []
        # for obs_id in (sorted(obs_ids)):
        #     slice_idx = self.include_history_steps - obs_id - 1
        #     start = -(self.skips * slice_idx + 1 + delay)* self.num_obs 
        #     end = -(self.skips * slice_idx + delay) * self.num_obs
        #     if end == 0:
        #         obs.append(self.obs_buf[:, start :])
        #     else:            
        #         obs.append(self.obs_buf[:, start : end])
        # return torch.cat(obs, dim=-1)

        obs = []
        for obs_id in (sorted(obs_ids)):
            slice_idx = self.include_history_steps - obs_id - 1
            start = -(self.skips * slice_idx + 1 + delay)* self.num_obs
            end = -(self.skips * slice_idx + delay) * self.num_obs
            end[end==0] = self.obs_buf.shape[1]
            
            
            # Create a range for slicing the obs_buf
            indices = torch.arange(self.num_obs, device=self.device).expand(self.obs_buf.shape[0], self.num_obs)
            obs_part = self.obs_buf[torch.arange(self.obs_buf.shape[0]).unsqueeze(1), end.unsqueeze(1) - self.num_obs + indices]
            
            obs.append(obs_part)
                
        return torch.cat(obs, dim=-1)


# device = 'cuda:0'
# num_obs=  2
# history_length = 5
# skips = 4
# obs_buf = ObservationBuffer(2, num_obs, history_length, skips, device=device)
# for i in range(20):
#     obs = torch.tensor([[2*i, 2*i+1], [2*i, 2*i+1]], device=device)
#     obs_buf.insert(obs)
# print(obs_buf.get_obs_vec(torch.arange(5)))
