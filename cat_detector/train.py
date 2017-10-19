import os
import sys
import cv2
import json
import torch
import pickle
import numpy as np
from tqdm import tqdm
from model import OwnModel
from IPython import embed
from torch.autograd import Variable
from data_provider import DataProvider


def train(args):
    if not os.path.isdir('./%s'%args.output):
        os.system('mkdir ./%s'%args.output)

    if args.cont:
        try:
            model = torch.load(args.cont)
            print('load success')
        except:
            model = OwnModel()
    else:
        model = OwnModel()
    dp = DataProvider(args.dataset)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    crit = torch.nn.CrossEntropyLoss()
    mbsize = 128
    for epoch in range(args.max_iter):
        odata, olabel = dp.train_iter(mbsize)
        data = Variable(torch.from_numpy(odata))
        label = Variable(torch.from_numpy(olabel))
        pred = model(data)
        pred = pred.contiguous().view(-1,2)
        loss = crit(pred, label)
        #optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('iter:%s loss:%s'%(epoch, loss.data.numpy()), end='\r')
        if epoch % 10 == 0:
            torch.save(model, './%s/model.pkl'%args.output)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--max-iter', default=100, type=int)
    parser.add_argument('--cont', default=None)
    parser.add_argument('--dataset')
    parser.add_argument('-o', '--output', default='train_log')
    args = parser.parse_args()

    train(args)
