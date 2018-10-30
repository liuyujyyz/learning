import numpy as np
from time import time
from decorators import timer

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

class UniformSampler:
    def __init__(self, N):
        self.list = list(range(N))
        self.size = N

    @timer
    def sample(self, m):
        if m > self.size:
            return None 
        c = self.size
        out = []
        for i in range(m):
            idx = np.random.randint(c)
            out.append(self.list[idx])
            self.list[idx] = self.list[c-1]
            self.list[c-1] = out[-1]
            c -= 1
        return out

if __name__ == '__main__':
    a = UniformSampler(100)
    b = a.sample(5)
    a = UniformSampler(10000000)
    b = a.sample(500000)
