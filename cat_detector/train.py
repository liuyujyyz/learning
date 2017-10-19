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
    if args.cont:
        try:
            model = torch.load(args.cont)
            print('load success')
        except:
            model = OwnModel()
    else:
        model = OwnModel()
    dp = DataProvider()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    mbsize = 128
    for epoch in tqdm(range(args.max_iter)):
        for minibatch in tqdm(range(50)):
            odata = dp.train_iter(mbsize)
            data = Variable(torch.from_numpy(odata))
            feature, label = model(data)
            mean = feature.mean(1)
            loss1 = ((feature-mean.expand_as(feature))**2).mean(1).mean()
            mean = label.mean(0)
            loss2 = ((label - mean.expand_as(label))**2).mean()
            loss = 0 - loss1 - loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if minibatch == 0:
                y = label.data.numpy()
                y = np.argmax(y, axis=1)
                for i in range(20):
                    img = odata[i].transpose(1,2,0)
                    pred = y[i]
                    cv2.imwrite('./tmp/%s_%s.jpg'%(i,pred), img)
        torch.save(model, 'model.pkl')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--max-iter', default=100, type=int)
    parser.add_argument('--cont', default=None)
    args = parser.parse_args()

    train(args)
