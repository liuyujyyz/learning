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
args = parser.parse_args()

model = torch.load(args.model)
L = list(os.walk(args.img))[0]
p,d,f = L
for item in f:
    img = cv2.imread(os.path.join(p, item))
    fimg = np.array([img.transpose(2,0,1)]).astype('float32')
    q = Variable(torch.from_numpy(fimg))
    out = model(q)
    out_img = out.data.numpy()[0,1,:,:]
    out_img -= out_img.min()
    out_img /= out_img.max()
    out_img *= 256
    out_img = out_img.astype('uint8')
    out_img = cv2.resize(out_img, (img.shape[1], img.shape[0]))
    out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)
    cv2.imshow('x', np.concatenate([img, out_img], axis=1))
    cv2.waitKey(0)
