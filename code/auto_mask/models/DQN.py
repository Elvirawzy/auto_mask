import torch.nn as nn
import torch.nn.functional as F

class DQNModel(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQNModel, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        self.sigmoid = nn.Sigmoid()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return self.sigmoid(x)
        # return x

class DQNCNNModel(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(DQNCNNModel, self).__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        num_linear_units = self.size_linear_unit * self.size_linear_unit * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)

        self.output = nn.Linear(in_features=128, out_features=num_actions)
        self.sigmoid = nn.Sigmoid()

    @property
    def size_linear_unit(self):
        size, kernel_size, stride = 10, 3,  1
        return (size - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.fc_hidden(x.view(x.size(0), -1)))
        x = self.output(x)
        # return self.sigmoid(x)
        return x