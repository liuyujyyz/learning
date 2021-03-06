import os
import sys
import json
import torch
import pickle
import numpy as np
from torch.autograd import Variable

class OwnModel(torch.nn.Module):
    def __init__(self):
        super(OwnModel, self).__init__()
        chans = [3,8,16,32,64,128,256]
        self.conv = [torch.nn.Conv2d(kernel_size = 3, stride = 1, padding = 1, out_channels=chans[i+1], in_channels=chans[i]) for i in range(5)]
        self.conv_in = [torch.nn.Conv2d(kernel_size = 3, stride = 1, padding = 1, out_channels=chans[i+1], in_channels=chans[i+1]) for i in range(5)]
        self.bn = [torch.nn.BatchNorm2d(chans[i+1]) for i in range(5)]
        self.bn_in = [torch.nn.BatchNorm2d(chans[i+1]) for i in range(5)]
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.outfc = torch.nn.Conv2d(kernel_size = 2, stride= 1, padding=0, in_channels=chans[5], out_channels=128)
        self.clser = torch.nn.Conv2d(kernel_size = 1, stride= 1, padding=0, in_channels=128, out_channels=2)
        self.softmax = torch.nn.Softmax2d()
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.chans = chans
        self.iter = 0

    def forward(self, x):
        self.iter += 1
        for i in range(5):
            x = self.bn[i](self.relu(self.conv[i](x)))
            if i >= 2:
                x = self.bn_in[i](self.relu(self.conv_in[i](x))) + x
            x = self.pool(x)
        q = self.relu(self.outfc(x))
        y = self.softmax(self.clser(q))
        return y

