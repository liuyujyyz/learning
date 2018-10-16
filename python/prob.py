import numpy as np
from time import time

class Distribution():
    def __init__(self, supp, prob):
        assert len(supp) == len(prob), 'not same length'
        supp = np.array(supp)
        prob = np.array(prob)
        prob = prob / prob.sum()

        index = np.argsort(supp)
        self.supp = supp[index]
        self.prob = prob[index]
        self.cdf = [0.0]
        for i in range(len(prob)-1):
            self.cdf.append(self.cdf[i] + self.prob[i])
        self.cdf = list(zip(self.supp, self.cdf))
        self.exp = None
        self.var = None
        self.length = len(self.supp)

    def _exp(self):
        if self.exp:
            return self.exp
        self.exp = (self.supp * self.prob).sum()
        return self.exp

    def _var(self):
        if self.var:
            return self.var
        tmp = self._exp()
        self.var = (((self.supp - tmp)**2)*self.prob).sum()
        return self.var

    def sample_one(self):
        #re = np.random.choice(range(self.length), p=self.prob)
        re = 0
        r = np.random.uniform(0,1)
        s = 0
        while s < r:
            s += self.prob[re]
            re += 1

        return self.supp[re - 1]

    def sample_one_v2(self):
        r = np.random.uniform(0,1)
        re = max(i for i,c in self.cdf if c <= r)
        return re

    def sample(self, leng):
        re = []
        for i in range(leng):
            re.append(self.sample_one_v2())
        return np.array(re)

    def show(self, item = 3):
        if item == 3:
            print(list(zip(self.supp, self.prob, self.cdf)))
        elif item == 2:
            print(list(zip(self.supp, self.prob)))

    def entropy(self):
        return -(self.prob * np.log(self.prob)).sum()

if __name__ == '__main__':
    N_sample = 100000
    for L in [10, 100, 1000, 10000]:
        dist = Distribution(np.random.randint(0,L*100,(L,)), np.random.uniform(0,1,(L,)))
        print(dist.entropy())
        start = time()    
        q = dist.sample(N_sample)
        print((time() - start) / (N_sample))
        unique, counts = np.unique(q, return_counts=True)
        m = Distribution(unique, counts)
        print(m.entropy())
