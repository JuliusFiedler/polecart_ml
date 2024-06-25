import numpy as np
from ipydex import IPS, activate_ips_on_exception
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PretrainedNet(nn.Module):
    def __init__(self):
        super(PretrainedNet, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
