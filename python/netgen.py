import numpy as np


def 

class RandGraph:
    def __init__(self, source=None):
        if source:
            self.source = source
        else:
            self.source = 'RY'

    def gen(self, node, **kwargs):
        V = []
        E = []
        if self.source == 'RY':
            assert 'threshold' in kwargs 
            V = list(range(node))
            prob = np.random.uniform(0, 1, (node, node))
            x, y = np.where(prob > kwargs['threshold'])
            E = list(zip(x,y))
            return (V, E)

if __name__ == '__main__':
    r = RandGraph()
    V, E = r.gen(4, threshold=0.5)
    print(E)


