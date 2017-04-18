
import numpy as np
import pickle

class DataProvider():
    def __init__(self, with_label):
        self.with_label = with_label

    def train(self, batch_size):
        batch_idx = 0
        length = self.train_img.shape[0]
        while True:
            idxs = np.arange(0, length)
            np.random.shuffle(idxs)
            for batch_idx in range(0, length, batch_size):
                cur_idx = idxs[batch_idx:batch_idx+batch_size]
                img = self.train_img[cur_idx]
                label = self.train_label[cur_idx]
                if self.with_label:
                    yield (img, label)
                else:
                    yield img

    def valid(self):
        if self.with_label:
            return (self.valid_img, self.valid_label)
        else:
            return self.valid_img

    def test(self):
        if self.with_label:
            return (self.test_img, self.test_label)
        else:
            return self.test_img


class MNIST(DataProvider):
    def __init__(self, with_label = True):
        a = pickle.load(open('../data/mnist.pkl','rb'), encoding='latin1')
        train, valid, test = a
        self.train_img, self.train_label = train
        self.valid_img, self.valid_label = valid
        self.test_img, self.test_label = test
        self.with_label = with_label

class cifar10(DataProvider):
    def __init__(self, with_label = True):
        self.with_label = with_label
        data = []
        label = []
        for i in range(5):
            a = pickle.load(open('../data/cifar-10-patches-py/data_batch_{}'.format(i+1), 'rb'), encoding='latin1')
            data.append(a['data'])
            label.append(a['labels'])
        data = np.concatenate(data, axis = 0)
        label = np.concatenate(label, axis = 0)
        self.train_img = data
        self.train_label = label
        a = pickle.load(open('../data/cifar-10-patches-py/test_batch','rb'), encoding='latin1')
        self.test_img = a['data']
        self.test_label = a['labels']
        self.valid_img = a['data'][::10]
        self.valid_label = a['labels'][::10]

