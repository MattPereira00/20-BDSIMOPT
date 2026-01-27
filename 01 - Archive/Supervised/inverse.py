
import torch.nn as nn

class InverseModel(nn.Module):
    def __init__(self, output_dim, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, y):
        return self.net(y)

