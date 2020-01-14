import numpy as np

class DataProvider():
    def __init__(self):
        self.c1 = np.array([[0.5,0.5], [0.5, -0.5], [-0.5, 0.5], [-0.5, -0.5]])
        self.c2 = np.array([[-0.2,-0.2], [0.2, 0.2], [0.2, -0.2], [-0.2, 0.2]])
        self.r1 = 0.01
        self.r2 = 0.03

    def get_data(self, batch=512):
        theta = np.random.uniform(0, 100, (batch//2, 1))
        delta = np.concatenate([np.cos(theta), np.sin(theta)], axis=1)
        tmp = np.random.randint(0, self.c1.shape[0], (batch//2,))
        a = np.zeros((batch//2, 2))
        a[:,0] = np.choose(tmp, self.c1[:,0])
        a[:,1] = np.choose(tmp, self.c1[:,1])

        tmp = np.random.randint(0, self.c2.shape[0], (batch//2,))
        b = np.zeros((batch//2, 2))
        b[:,0] = np.choose(tmp, self.c2[:,0])
        b[:,1] = np.choose(tmp, self.c2[:,1])

        data = np.concatenate([a+delta*self.r1, b+delta*self.r2], axis=0)
        label = np.zeros((512, ), dtype='uint8')
        label[256:] = 1
        return data, label


class DataProvider2():
    def __init__(self):
        self.r1 = 0.2
        self.r2 = 0.6

    def get_data(self, batch=512):
        r = np.random.uniform(0, 1, (batch, 1))
        theta = np.random.uniform(0,100, (batch, 1))
        point = np.concatenate([r*np.cos(theta), r*np.sin(theta)], axis=1)
        label = (r > 0.2)
        return point, label[:,0]
