import numpy as np
from prob import Distribution, Markov
from decorators import timer

def softmax(L):
    a = np.array(L)
    a = a - a.max()
    a = np.exp(a)
    a = a / a.sum()
    return list(a)

class MAB:
    def __init__(self, arm_list):
        self.arms = arm_list
        self.size = len(arm_list)

    def next(self):
        out = [arm.next() for arm in self.arms]
        return out

    def next_one(self, i):
        out = self.arms[i].next()
        return out

class MABChooser:
    def __init__(self, value=lambda x:x):
        self.valueFunc = value
        self.strategy = None

    @timer
    def learn(self, mab: MAB):
        supp = list(range(mab.size))
        prob = np.zeros((mab.size,), dtype='float32')
        std = np.zeros((mab.size,), dtype='float32')
        candidate = np.ones((mab.size,), dtype='uint8')
        s = 1000
        r = 1
        while candidate.sum() > 1:
            for j in np.where(candidate>0.5)[0]:
                samples = []
                for i in range(s):
                    samples.append(self.valueFunc(mab.next_one(j)))
                prob[j] = prob[j] * 0.2 + np.sum(samples) * 0.8
                std[j] = std[j] * 0.2 + np.std(samples) * 0.8
            med = np.median((prob)[candidate>0.5])
            # med = np.median((prob-3*std)[candidate>0.5])
            candidate[prob + 3*std < med] = 0
            s = int(s*1.2)

        self.strategy = Distribution(supp, softmax(prob))
    
    def play(self, mab: MAB):
        if self.strategy is None:
            self.learn(mab)
        out = 0
        best = 0
        tmp = mab.next()
        values = np.array([self.valueFunc(i) for i in tmp])
        arm_cum = np.zeros((mab.size,), dtype=values.dtype)
        for i in range(1000):
            tmp = mab.next()
            values = np.array([self.valueFunc(i) for i in tmp])
            choice = self.strategy.sample_one()
            out += values[choice]
            best += max(values)
            arm_cum += values

        return (out/best, out/arm_cum.max())
        
if __name__ == '__main__':
    dists = [Distribution(np.random.randint(10, 100, (10,)), np.random.randint(1, 100, (10,))) for i in range(100)]
    maxkov = [Markov(np.random.randint(10, 100, (10,)), np.random.randint(1, 100, (10, 10)), np.random.randint(0,10)) for i in range(100)]
    mab = MAB(dists+maxkov)
    """
    h = MABChooser(value = lambda x: x**2)
    h.learn(mab)
    h.play(mab)
    """
    h2 = MABChooser(value = lambda x: (x-25)**2)
    h2.learn(mab)
    r1, r2 = h2.play(mab)
    print(r1, r2)
