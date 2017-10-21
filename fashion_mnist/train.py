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

def get_lr(k):
    if k <= 30:
        return 0.1
    if k <= 100:
        return 0.01
    if k <= 300:
        return 0.001
    return 0.0001
    if k <= 300:
        mx = 1 - 0.9 / 30 * k
    elif k <= 400:
        mx = 1 - 0.09 / 10 * (k-300)
    elif k <= 500:
        mx = 1 - 0.009 / 10 * (k - 400)
    elif k <= 600:
        mx = 1 - 0.0009 / 10 * (k - 500)
    else:
        mx = 1e-4

    q = k % 100
    st, ed = 1e-6, mx
    if q > 50:
        st, ed = ed, st
        q = q - 50
    return (st + q * (ed - st) / 50) * 0.01


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
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    crit = torch.nn.CrossEntropyLoss()
    mbsize = 1024
    for epoch in range(args.max_iter):
        odata, olabel = dp.train_iter(mbsize)
        data = Variable(torch.from_numpy(odata))
        label = Variable(torch.from_numpy(olabel))
        lr = get_lr(model.iter)
        pred = model(data)
        pred = pred.contiguous().view(-1,2)
        loss = crit(pred, label)
        optimizer.zero_grad()
        for group in optimizer.param_groups:
            group['lr'] = lr
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
