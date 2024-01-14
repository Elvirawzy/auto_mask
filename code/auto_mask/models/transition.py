import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class DeterministicTransitionModel(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.fc = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.ln = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, obs_dim)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(self.ln(x))
        mu = self.fc_mu(x)
        return mu


class TransCNN(nn.Module):

    def __init__(self, channel, obs_dim, act_dim):
        super().__init__()
        self.channel = channel
        # self.conv = nn.Conv2d(channel, 16, kernel_size=3, stride=1)
        #
        # num_linear_units = self.size_linear_unit * self.size_linear_unit * 16
        # self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)
        #
        # self.fc = nn.Linear(in_features=128 + act_dim, out_features=128)
        # self.output = nn.Linear(in_features=128, out_features=10 * 10 * (channel+1))
        self.fc1 = nn.Linear(obs_dim+act_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out_fc = nn.Linear(128, obs_dim)

    @property
    def size_linear_unit(self):
        size, kernel_size, stride = 10, 3,  1
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        # x = F.relu(self.conv(x))
        # x = F.relu(self.fc_hidden(x.view(x.size(0), -1)))
        # x = torch.concatenate([x, a], dim=1)
        # x = F.relu(self.fc(x))
        # return self.output(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out_fc(x)
        return x

    def pred_s(self, x):
        logits = self(x)
        # logits = logits.view(-1, 10, 10, (self.channel+1))
        categoricals = Categorical(logits=logits)
        x_ = categoricals.sample()
        logprob = categoricals.log_prob(x_)
        return x_, logprob