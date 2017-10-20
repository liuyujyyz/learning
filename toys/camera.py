import torch
import numpy as np
import cv2
import torch.nn as nn
from torch.autograd import Variable
cv2.namedWindow('mw')

vid = cv2.VideoCapture(0)
buff = []

while True:
    try:
        conv = [nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1) for i in range(5)]
        relu = nn.ReLU()
        relu2 = nn.Sigmoid()

        while vid.isOpened():
            ret, frame = vid.read()
            if ret:
                buff.append(frame)
                if len(buff) > 5:
                    buff.pop(0)
                tmp = np.array(buff).astype('float32').mean(axis=0, keepdims=True).transpose((0,3,1,2))
                #tmp = np.array([frame], dtype='float32').transpose((0,3,1,2))
                x = Variable(torch.from_numpy(tmp))
                #relu
                h = conv[0](x)
                h = relu(h)
                h = conv[1](x)
                h = relu(h)
                h = h.data.numpy()[0]
                h = h.transpose(1,2,0)
                #swish
                y = conv[0](x)
                y = 1.67653251702 * (y * relu2(y) - 0.20662096414)
                y = conv[1](y)
                y = 1.67653251702 * (y * relu2(y) - 0.20662096414)
                y = y.data.numpy()[0]
                y = y.transpose(1,2,0)
                def convert(z):
        #            z = ((z - z.min())/(z.max()-z.min())*frame)
                    z = ((z - z.min())/(z.max()-z.min())*256).astype('uint8')
                    return z
                cv2.imshow('mw', np.concatenate([convert(h), convert(y)], axis=1))
                key = cv2.waitKey(10) & 0xff
                if key == ord('s'):
                    raise ValueError()
                if key == ord('q'):
                    raise TypeError()
    except ValueError:
        continue
    except TypeError:
        break
