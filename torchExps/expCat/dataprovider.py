import numpy as np
import os
import cv2 
import pickle

class DataProvider():
    def __init__(self):
        path = '/home/liuyu/liuyu_lib/learning/data/cat'
        self.traindata = []
        self.trainlabel = []
        self.testdata = []
        self.testlabel = []
        for i in range(10):
            L = os.walk(os.path.join(path, str(i)))
            p, d, f = next(L)
            label = i
            size = len(f)
            A = int(size*0.9)
            for j in range(size):
                img = cv2.imread(os.path.join(p, f[j]))
                if img is None:
                    continue
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img = cv2.resize(img, (128, 128))
                if j < A:
                    self.traindata.append(img)
                    self.trainlabel.append(label)
                else:
                    self.testdata.append(img)
                    self.testlabel.append(label)
        self.traindata = np.array(self.traindata)
        self.trainlabel = np.array(self.trainlabel)
        self.testdata = np.array(self.testdata)
        self.testlabel = np.array(self.testlabel)
                

    def get_data(self, batch=512):
        while True:
            index = np.random.permutation(self.traindata.shape[0])
            data = self.traindata[index]
            label = self.trainlabel[index]
            i = 0
            while True:
                st = i * batch
                ed = st + batch
                if ed > data.shape[0]:
                    break
                img = data[st:ed]
                img = img + np.random.uniform(0, 10, img.shape)
                img = (img - 128.0) / 128.0
                batchData = img.transpose((0,3,1,2))
                batchLabel = label[st:ed]

                yield batchData, batchLabel

    def get_test_data(self, batch=512):
        while True:
            index = np.random.permutation(self.testdata.shape[0])
            data = self.testdata[index]
            label = self.testlabel[index]
            i = 0
            while True:
                st = i * batch
                ed = st + batch
                if ed > data.shape[0]:
                    break
                img = data[st:ed]
                img = (img - 128.0) / 128.0
                batchData = img.transpose((0,3,1,2))
                batchLabel = label[st:ed]

                yield batchData, batchLabel

