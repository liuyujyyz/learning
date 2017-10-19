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
        chans = [3,4,8,16,32,64,128,256]
        self.conv = [torch.nn.Conv2d(kernel_size = 3, stride = 2, padding = 1, out_channels=chans[i+1], in_channels=chans[i]) for i in range(5)]
        self.bn = [torch.nn.BatchNorm2d(chans[i+1]) for i in range(5)]
        
        self.outfc = torch.nn.Linear(chans[5]*2*2, 128)
#        self.outfc = torch.nn.Linear(3*64*64, 128)
        self.clser = torch.nn.Linear(128, 2)
        self.softmax = torch.nn.Softmax()
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.chans = chans

    def forward(self, x):
        for i in range(5):
            x = self.bn[i](self.tanh(self.conv[i](x)))
            #x = self.bn[i](self.tanh(self.conv[i](x)))
#            print(x.data.numpy().shape)
        q = x.contiguous().view(-1, self.chans[5]*2*2)
        q = self.relu(self.outfc(q))
        y = self.softmax(self.clser(q))
        return q, y

