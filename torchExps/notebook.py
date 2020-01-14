import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
import cv2
import imageio


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 6)
        self.fc2 = nn.Linear(6, 6)
        self.fc3 = nn.Linear(6, 2)

    def forward(self, x):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(y1))
        y3 = self.fc3(y2)
        out = F.log_softmax(y3, dim=1)
        return out


def train(model, optimizer, epoch, device):
    model.train()
    label = np.zeros((512, ), dtype='uint8')
    label[256:] = 1
    target = torch.tensor(label, device=device).long()

    for batch in range(500):
        data = torch.tensor(make_data(), device=device).float()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def make_data():
    c1 = np.array([[0.5,0.5]])
    c2 = np.array([[-0.5,-0.5]])
    c3 = np.array([[0.5, -0.5]])
    c4 = np.array([[0.5, 0]])
    r1 = 0.1
    r2 = 0.3
    theta = np.random.uniform(0, 100, (256, 1))
    delta = np.concatenate([np.cos(theta), np.sin(theta)], axis=1)
    if np.random.uniform(0,1) > 0.5:
        a = c1
    else:
        a = c3
    if np.random.uniform(0,1) > 0.5:
        b = c2
    else:
        b = c4

    data = np.concatenate([a+delta*r1, b+delta*r2], axis=0)
    return data


class GIFWriter:
    def __init__(self):
        self.images = []
        self.H = None
        self.W = None
        self.channel = None

    def save(self, filename, fps=30, loop=0):
        with imageio.get_writer(filename, mode='I', fps=fps, loop=loop) as writer:
            for image in tqdm(self.images):
                writer.append_data(image)

    def append(self, image):
        shape = image.shape
        if len(shape) == 2:
            shape = tuple(shape) + (1,)
        if len(self.images) == 0:
            self.W, self.H, self.channel = shape
        else:
            W, H, C = shape
            assert (W==self.W) and (H==self.H) and (C==self.channel)

        self.images.append(image)


def test(model, epoch):
    img = np.ones((512, 512, 3), dtype='uint8') * 255
    cv2.circle(img, (128, 128), int(256*0.3), (128, 128, 128), -1)
    cv2.circle(img, (384, 384), int(256*0.1), (128, 0, 0), -1)
    cv2.circle(img, (128, 384), int(256*0.1), (128, 0, 0), -1)
    cv2.circle(img, (256, 384), int(256*0.3), (128, 128, 128), -1)
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

    U = GIFWriter()

    for epoch in tqdm(range(100)):
        train(model, optimizer, epoch, device)
        schLR.step()
        img = test(model, epoch)
        U.append(img)
        
    U.save('test.gif', fps=5, loop=-1)


if __name__ == '__main__':
    main()
