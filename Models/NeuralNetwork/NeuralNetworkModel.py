import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNetwork(nn.Module):
    def __init__(self, in_channels = 26, out_channels = 100):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 50)  # Categories + Metrics
        # self.fc1 = nn.Linear(19, 128) # Categories Only
        self.fc2 = nn.Linear(50, out_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
"""
super().__init__()
        self.fc1 = nn.Linear(in_channels, 128)  # Categories + Metrics
        # self.fc1 = nn.Linear(19, 128) # Categories Only
        self.fc2 = nn.Linear(128, 2048)
        self.fc3 = nn.Linear(2048, out_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x
"""