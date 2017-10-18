
import numpy as np
import pickle

class DataProvider():
    def __init__(self, with_label, label_size):
        self.with_label = with_label
        self.label_size = label_size

    def train(self, batch_size):
        batch_idx = 0
        length = self.train_img.shape[0]
        while True:
            idxs = np.arange(0, length)
            np.random.shuffle(idxs)
            for batch_idx in range(0, length, batch_size):
                cur_idx = idxs[batch_idx:batch_idx+batch_size]
                img = self.train_img[cur_idx]
                if np.random.uniform(0,1) > 0.5:
                   img += np.random.normal(0,1e-2,img.shape)

                if self.with_label:
                    label = self.train_label[cur_idx]
                    ohl = np.zeros((label.shape[0], self.label_size))
                    ohl[range(label.shape[0]), label] += 1
                    yield (img, ohl)
                else:
                    yield img

    def valid(self):
        if self.with_label:
            size_v = self.valid_label.shape[0]
            valid = np.zeros((size_v, self.label_size))
            valid[range(size_v), self.valid_label] += 1
            return (self.valid_img, valid)
        else:
            return self.valid_img

    def test(self):
        if self.with_label:
            size_t = self.test_label.shape[0]
            test = np.zeros((size_t, self.label_size))
            test[range(size_t), self.test_label] += 1
            return (self.test_img, test)
        else:
            return self.test_img


class MNIST(DataProvider):
    def __init__(self, with_label = True, label_size = 10):
        self.label_size = label_size
        a = pickle.load(open('data/mnist.pkl','rb'), encoding='latin1')
        train, valid, test = a
        self.train_img, self.train_label = train
        self.valid_img, self.valid_label = valid
        self.test_img, self.test_label = test
        self.with_label = with_label

class cifar10(DataProvider):
    def __init__(self, with_label = True, label_size = 10):
        self.with_label = with_label
        self.label_size = label_size
        data = []
        label = []
        for i in range(5):
            a = pickle.load(open('data/cifar-10-patches-py/data_batch_{}'.format(i+1), 'rb'), encoding='latin1')
            data.append(a['data'])
            label.append(a['labels'])
        data = np.concatenate(data, axis = 0)
        label = np.concatenate(label, axis = 0)
        self.train_img = data
        self.train_label = label
        a = pickle.load(open('data/cifar-10-patches-py/test_batch','rb'), encoding='latin1')
        self.test_img = a['data']
        self.test_label = a['labels']
        self.valid_img = a['data'][::10]
        self.valid_label = a['labels'][::10]

