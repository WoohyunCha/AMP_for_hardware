import numpy as np
import torch
import torch.nn as nn

class Morphnet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256]):
        super(Morphnet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.linear1 = nn.Linear(input_dim, hidden_dims[0])
        self.linear2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.head = nn.Linear(hidden_dims[1], output_dim)
        self.activation1 = nn.Tanh()
        self.activation2 = nn.Tanh()



    def forward(self, x: torch.Tensor):
        # Input shape: (B, d, T) = (B, 8, 50)
        x = x.flatten(start_dim=1)
        x = self.linear1(x)
        x = self.activation1(x)

        x = self.linear2(x)
        x = self.activation2(x)

        return self.head(x)
    
    # def to(self, *args, **kwargs):
    #     # Call the base class method to move parameters and registered buffers
    #     model = super(CNNEncoder, self).to(*args, **kwargs)

    #     # Manually move obs_buf to the new device
    #     self.obs_buf = self.obs_buf.to(next(model.parameters()).device)

    #     return model
