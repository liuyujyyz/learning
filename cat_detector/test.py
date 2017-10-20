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
                if out < args.thr:
                    re.append([(i/qimg.shape[0]*img.shape[0],j/qimg.shape[1]*img.shape[1]), ((i+64)/qimg.shape[0]*img.shape[0], (j+64)/qimg.shape[1]*img.shape[1]), out])
    re.sort(key=lambda x:x[1])
    print(len(re))
    for item in re:
        s,t,_ = item
        cv2.rectangle(img, (int(s[1]), int(s[0])), (int(t[1]), int(t[0])), (0,0,255), 2)
    cv2.imshow('x', img)
    cv2.waitKey(0)
