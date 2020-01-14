import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
from dataprovider import *
from model import Net
import cv2

def train(model, dp, optimizer, epoch, device):
    model.train()

    for batch in range(500):
        data, target = dp.get_data()
        target = torch.tensor(target, device=device).long()
        data = torch.tensor(data, device=device).float()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

def test(model, epoch, dp):
    img = np.ones((512, 512, 3), dtype='uint8') * 255
    cv2.circle(img, (256, 256), int(dp.r1*256), (128, 128, 128), 2)
    cv2.circle(img, (256, 256), int(dp.r2*256), (128, 128, 128), 2)
    cv2.putText(img, str(epoch), (100,100), cv2.FONT_HERSHEY_PLAIN, 10, (0,0,0)) 
    model.eval()
    with torch.no_grad():
        for i in range(512):
            data = ((np.array(range(512)) - 256.0)/256.0).reshape((512, 1))
            data = np.concatenate([np.ones((512, 1))*(i-256.0)/256.0, data], axis=1)
            data = torch.tensor(data).float()
            output = model(data)
            tmp = output.data.numpy()
            img[i,tmp[:,0]>tmp[:,1],0] -= 128

    return img


def main():
    device = torch.device('cpu')
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8, weight_decay=0.01)
    schLR = StepLR(optimizer, step_size=1, gamma=0.9)
    dp = DataProvider2()
    from utils import GIFWriter
    U = GIFWriter()

    for epoch in tqdm(range(100)):
        train(model, dp, optimizer, epoch, device)
        schLR.step()
        img = test(model, epoch, dp)
        U.append(img)
        
    U.save('4points.gif', fps=5, loop=-1)

if __name__ == '__main__':
    main()

