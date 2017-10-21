import os
import sys
import cv2
import json
import torch
import pickle
import numpy as np
from torch.autograd import Variable

class DataProvider():
    def __init__(self, dataset):
        data = pickle.load(open(dataset,'rb'))
        self.imgs = list(data.keys())
        self.bbox = [data[key] for key in self.imgs]
        self.size = len(self.imgs)

    def crop(self, img, bbox):
        if len(bbox) == 0:
            pos = None
        else:
            idx = np.random.randint(len(bbox))
            tmp = bbox[idx]
            s_x,s_y = tmp[0]
            t_x,t_y = tmp[1]
            center_x = (s_x+t_x)/np.random.uniform(1.99,2.01)
            center_y = (s_y+t_y)/np.random.uniform(1.99,2.01)
            L_x = np.abs(s_x - center_x)
            L_y = np.abs(s_y - center_y)
            ub = min(img.shape[1] - center_x, img.shape[0] - center_y)
            L = max(L_x, L_y)
            ratio = np.random.uniform(1,1.2)
            L = L * ratio
            L = min(L, ub)
            pts1 = np.float32([[center_x-L, center_y-L],[center_x+L, center_y-L],[center_x-L, center_y+L]])
            pts2 = np.float32([[0,0],[64,0],[0,64]])
            mat = cv2.getAffineTransform(pts1, pts2)
            pos = cv2.warpAffine(img, mat, (64,64))
        neg = None
        cop = img.copy()
        for item in bbox:
            cv2.rectangle(cop, item[0], item[1], (128,128,128), -1)
        s_x = np.random.randint(img.shape[1])
        s_y = np.random.randint(img.shape[0])
        L = np.random.randint(min(img.shape[1]-s_x, img.shape[0]-s_y))
        pts1 = np.float32([[s_x,s_y],[s_x+L,s_y],[s_x,s_y+L]])
        pts2 = np.float32([[0,0],[64,0],[0,64]])
        mat = cv2.getAffineTransform(pts1, pts2)
        neg = cv2.warpAffine(cop, mat, (64,64))
        return pos, neg

    def train_iter(self, mbsize):
        re = []
        L = []
        while len(re) < mbsize:
            idx = np.random.randint(self.size)
            img = cv2.imread(self.imgs[idx])
            bbox = self.bbox[idx]
            pos, neg = self.crop(img, bbox)
            if not(pos is None):
                if np.random.uniform(0,1) > 0.5:
                    pos = pos[:,::-1,:]
                re.append(pos)
                L.append(1)
            if not(neg is None) and np.sum(L)*3 > len(L):
                re.append(neg)
                L.append(0)
        return np.array(re[:mbsize]).transpose(0,3,1,2).astype('float32'), np.array(L[:mbsize])





