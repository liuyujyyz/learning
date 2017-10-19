import os
import sys
import cv2
import json
import torch
import pickle
import numpy as np
from torch.autograd import Variable

class DataProvider():
    def __init__(self):
        try:
            data = pickle.load(open('../data/cat.pkl','rb'))
        except:
            L = list(os.walk('../data/cat'))[0]
            p,d,f = L
            data = {'imgs':[], 'bbox':[]}
            for item in f:
                img = cv2.imread(os.path.join(p, item))
                if img is None:
                    print(item)
                    continue
                data['imgs'].append(img)
                data['bbox'].append(None)
            pickle.dump(data, open('../data/cat.pkl','wb'))
        self.imgs = data['imgs']
        self.bbox = data['bbox']
        self.size = len(self.imgs)

    def train_iter(self, mbsize):
        re = []
        for i in range(mbsize):
            idx = np.random.randint(self.size)
            img = self.imgs[idx]
            s_x = np.random.randint(img.shape[0] - 30)
            s_y = np.random.randint(img.shape[1] - 30)
            L = np.random.randint(min(img.shape[0]-30-s_x, img.shape[1]-30-s_y)) + 30
            crop = img[s_x:s_x+L,s_y:s_y+L,:]
            single = cv2.resize(crop, (64,64))
            re.append(single)
        return np.array(re).transpose(0,3,1,2).astype('float32')





