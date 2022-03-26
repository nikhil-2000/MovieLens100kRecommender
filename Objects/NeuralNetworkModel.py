import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNetwork(nn.Module):
    def __init__(self, in_channels = 24, out_channels = 2048):

        super().__init__()
        self.fc1 = nn.Linear(in_channels, 128) # Categories + Metrics
        # self.fc1 = nn.Linear(19, 128) # Categories Only
        self.fc2 = nn.Linear(128, 9216)
        self.fc3 = nn.Linear(9216, 4608)
        self.fc4 = nn.Linear(4608, out_channels)


    def forward(self, x):
        assert not torch.any(torch.isnan(x))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)

        return x