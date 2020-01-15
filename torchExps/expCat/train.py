import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm, trange
from dataprovider import *
from model import Net
import cv2

def train(model, dp, optimizer, epoch, device):
    model.train()
    t = trange(50, desc='bar desc', leave=True)
    flag = False

    for batch in t:
        data, target = next(dp)
        target = torch.tensor(target, device=device).long()
        data = torch.tensor(data, device=device).float()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if loss.item() < 1:
            flag = True
        t.set_description('loss: %s' % str(loss.item()))
        t.refresh()
    return flag

def test(model, epoch, dp):
    model.eval()
    with torch.no_grad():
        data, label = next(dp)
        output = model(torch.tensor(data).float())
        tmp = output.data.numpy()
        for i in range(10):
            index = tmp[i].argmax()
            if index != label[i]:
                print(index, label[i], tmp.max())
                cv2.imshow('x', (data[i]*128+128).transpose((1,2,0)).astype('uint8'))
                cv2.waitKey(0)
                

def main():
    device = torch.device('cpu')
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.8, weight_decay=1e-4)
    schLR = StepLR(optimizer, step_size=1, gamma=0.99)
    dp = DataProvider()
    from utils import GIFWriter
    U = GIFWriter()
    dg = dp.get_data()
    tg = dp.get_test_data(batch=10)

    for epoch in tqdm(range(100)):
        good = train(model, dg, optimizer, epoch, device)
        schLR.step()
        if good:
            test(model, epoch, tg)

    torch.save(model.state_dict, 'cat_classifier.pt')

if __name__ == '__main__':
    main()

