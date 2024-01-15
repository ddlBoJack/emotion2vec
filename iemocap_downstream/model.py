import torch
from torch import nn

class BaseModel(nn.Module):
    def __init__(self, input_dim=768, output_dim=4):
        super().__init__()
        self.pre_net = nn.Linear(input_dim, 256)

        self.post_net = nn.Linear(256, output_dim)
        
        self.activate = nn.ReLU()

    def forward(self, x, padding_mask=None):
        x = self.activate(self.pre_net(x))

        x = x * (1 - padding_mask.unsqueeze(-1).float())
        x = x.sum(dim=1) / (1 - padding_mask.float()
                            ).sum(dim=1, keepdim=True)  # Compute average
        
        x = self.post_net(x)
        return x
