import os,sys
import cv2
import json, pickle
import numpy as np
from model import OwnModel
from torch.autograd import Variable
import torch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--img')
parser.add_argument('--thr', type=float, default=0.5)
args = parser.parse_args()

model = torch.load(args.model)
L = list(os.walk(args.img))[0]
p,d,f = L
for item in f:
    re = []
    img = cv2.imread(os.path.join(p, item))
    ratio = img.shape[1] / img.shape[0]
    for target in [64,128,256,512,1024]:
        qimg = cv2.resize(img, (int(target * ratio), target))
        for i in range(0, qimg.shape[0]-64, 32):
            for j in range(0, qimg.shape[1]-64, 32):
                tmp = np.array(qimg[i:i+64, j:j+64,:])
                fimg = np.array([tmp.transpose(2,0,1)]).astype('float32')
                q = Variable(torch.from_numpy(fimg))
                out = model(q)
                out = out.data.numpy()[0,1,0,0]
                if out > args.thr:
                    re.append([tmp, out])
    re.sort(key=lambda x:x[1])
    for i in range(10):
        tmp = re[-1-i][0]
        cv2.imshow('x', tmp)
        cv2.waitKey(0)
