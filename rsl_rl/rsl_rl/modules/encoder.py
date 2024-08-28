import numpy as np
import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, history_length=50):
        super(CNNEncoder, self).__init__()

        # Layer 1: Conv1D with kernel size 6, 32 filters, and stride 3
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=6, stride=3, padding=0)
        self.relu1 = nn.ReLU()

        # Layer 2: Conv1D with kernel size 4, 16 filters, and stride 2
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=0)
        self.relu2 = nn.ReLU()

        # Adaptive average pooling to aggregate the time dimension
        # self.pooling = nn.AdaptiveAvgPool1d(1)

        # Fully connected layer to produce output of shape (B, output_dim)
        # self.fc1 = nn.Linear(16, output_dim)  # Adjusted input size to match output of pooling
        self.fc2 = nn.Linear(16*6, output_dim)
        # self.num_envs = num_envs
        # self.num_obs = input_dim
        # self.include_history_steps = history_length
        # self.skips = 10
        # self.num_obs_total = input_dim * ((self.include_history_steps-1) * self.skips + 1)

        # self.register_buffer('obs_buf', torch.zeros(self.num_envs, self.num_obs_total, dtype=torch.float))
        # self.obs_buf = torch.zeros(self.num_envs, self.num_obs_total, dtype=torch.float)



    def forward(self, x):
        # Input shape: (B, d, T) = (B, 8, 50)
        
        # Apply first convolutional layer
        x = self.conv1(x)  # Output shape: (B, 32, 15)
        x = self.relu1(x)

        # Apply second convolutional layer
        x = self.conv2(x)  # Output shape: (B, 16, 6)
        x = self.relu2(x)

        # Apply pooling across the time dimension (adaptive average pooling)
        # x = self.pooling(x)  # Output shape: (B, 16, 1)
        # x = torch.squeeze(x, -1)
        # Remove the time dimension, resulting in shape (B, 16)
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers
        # x = self.fc1(x)  # Output shape: (B, 128)
        x = self.fc2(x)
        return x
    
    # def to(self, *args, **kwargs):
    #     # Call the base class method to move parameters and registered buffers
    #     model = super(CNNEncoder, self).to(*args, **kwargs)

    #     # Manually move obs_buf to the new device
    #     self.obs_buf = self.obs_buf.to(next(model.parameters()).device)

    #     return model
