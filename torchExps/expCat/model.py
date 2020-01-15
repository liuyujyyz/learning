import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv0 = nn.Conv2d(3, 16, 3, 2)
        self.conv1 = nn.Conv2d(16, 32, 3, 2)
        self.conv11 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2)
        self.conv22 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2)
        self.conv33 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc0 = nn.Linear(128, 64)
        self.fc1 = nn.Linear(64, 10)

    def forward(self, x):
        y0 = F.relu(self.conv0(x))
        y0 = (y0 - y0.mean())/y0.std()

        y1 = F.relu(self.conv1(y0))
        y1 = (y1 - y1.mean())/y1.std()
        y1 = y1 + F.relu(self.conv11(y1))

        y2 = F.relu(self.conv2(y1))
        y2 = (y2 - y2.mean())/y2.std()
        y2 = y2 + F.relu(self.conv22(y2))

        y3 = F.relu(self.conv3(y2))
        y3 = (y3 - y3.mean())/y3.std()
        y3 = y3 + F.relu(self.conv33(y3))

        y4 = self.gap(y3)

        y4 = y4.reshape(y4.shape[:2])
        y4 = F.relu(self.fc0(y4))

        y5 = self.fc1(y4)
        out = F.softmax(y5, dim=1)
        return out



