import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.fc = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        rwd = self.fc3(x)
        return rwd


class RewardCNN(nn.Module):
    def __init__(self, channel, act_dim):
        super().__init__()
        self.conv = nn.Conv2d(channel, 16, kernel_size=3, stride=1)

        num_linear_units = self.size_linear_unit * self.size_linear_unit * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)

        self.fc = nn.Linear(in_features=128 + act_dim, out_features=128)
        self.output = nn.Linear(in_features=128, out_features=1)

    @property
    def size_linear_unit(self):
        size, kernel_size, stride = 10, 3,  1
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x, a):
        x = F.relu(self.conv(x))
        x = F.relu(self.fc_hidden(x.view(x.size(0), -1)))
        x = torch.concatenate([x, a], dim=1)
        x = F.relu(self.fc(x))
        return self.output(x)

    def pred_r(self, x, a):
        mu = self(x, a)
        return mu
