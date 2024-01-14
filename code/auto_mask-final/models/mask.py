import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical

class MaskModel(nn.Module):
    def __init__(self, obs_dim, action_dim, load_model):
        super(MaskModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim*2)
        )
        self.action_num = action_dim
        self.load_model = load_model

    def forward(self, x):
        x = self.fc(x)
        return x

    def get_mask(self, x, mask=None):
        logits = self.forward(x).view(-1, self.action_num, 2)
        categoricals = Categorical(logits=torch.clamp(logits, 0., 1.))
        if mask is None:
            mask = categoricals.sample()
        logprob = categoricals.log_prob(mask)
        return mask, logprob


class GlobalMask(nn.Module):
    def __init__(self, act_dim, device, load_model):
        super().__init__()
        probs = torch.concatenate([torch.zeros(act_dim, 1), torch.ones(act_dim, 1)], dim=1)
        self.probs = torch.nn.Parameter(probs.clone().to(torch.float32).to(device))
        self.load_model = load_model

    def forward(self):
        pass

    def get_mask(self, m=None):
        if self.load_model:
            categoricals = Categorical(probs=torch.round(self.probs))
        else:
            categoricals = Categorical(probs=torch.clamp(self.probs, 0., 1.))
        if m is None:
            m = categoricals.sample()
        # logprob = categoricals.log_prob(m)
        return m
