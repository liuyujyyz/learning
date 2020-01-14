import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 6)
        self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6, 2)

    def forward(self, x):
        y1 = F.tanh(self.fc1(x))
        y2 = F.tanh(self.fc2(y1))
        y3 = self.fc3(y2)
        out = F.log_softmax(y3, dim=1)
        return out



