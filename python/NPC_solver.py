import numpy as np
from structure import Graph


class SubmodularSolver:
    def __init__(self, elements, func):
        self.elements = elements
        self.func = func

    def solve(self, constraint, K):
        M = self.elements.copy()
        solution = []
        while True:
            best = -1e10
            add = None
            baseline = self.func(solution)
            baseconst = constraint(solution)
            for item in M:
                tmp = solution + [item]
                U = constraint(tmp)
                if U > K:
                    continue
                delta = (self.func(tmp) - baseline) / (U - baseconst)
                if delta > best:
                    best = delta
                    add = item
            if add is not None:
                solution = solution + [add]
                M.remove(add)
            else:
                break
        return solution


class KnapsackSolver:
    def __init__(self, weights, values):
        ratio = values / weights
        index = np.argsort(ratio)[::-1]
        self.weights = weights[index]
        self.values = values[index]

    def find_min_w(self, preserve = None):
        if preserve is None:
            return np.argmin(self.weights)
        else:
            best = -1
            minw = 10000000
            for i in range(len(self.weights)):
                if i in preserve:
                    continue
                if self.weights[i] < minw:
                    minw = self.weights[i] 
                    best = i
            return best

    def find_max_v(self, preserve = None):
        if preserve is None:
            return np.argmax(self.values)
        else:
            best = -1
            maxw = -1
            for i in range(len(self.values)):
                if i in preserve:
                    continue
                if self.values[i] > maxw:
                    maxw = self.values[i]
                    best = i
            return best

    def greedy(self, capacity, preserve = None):
        w = 0
        v = 0
        item = []
        if preserve:
            for q in preserve:
                w += self.weights[q]
                v += self.values[q]
            item += preserve

        deltaw = 0
        deltav = 0
        deltaitem = []
        for i in range(len(self.weights)):
            if i in item:
                continue
            if w + deltaw + self.weights[i] <= capacity:
                deltaw += self.weights[i]
                deltav += self.values[i]
                deltaitem.append(i)
        idx = self.find_max_v(preserve)
        if self.weights[idx] + w <= capacity and self.values[idx] > deltav:
            deltaw = self.weights[idx]
            deltav = self.values[idx]
            deltaitem = [idx]
        w += deltaw
        v += deltav
        item += deltaitem
        minw = self.weights[self.find_min_w(item)] 
        if minw + w <= capacity:
            return self.greedy(capacity, item)
        else:
            return v, item

    def total_w(self, index):
        index = np.array(index)
        return self.weights[index].sum()

    def total_v(self, index):
        index = np.array(index)
        return self.values[index].sum()


if __name__ == '__main__':
    # np.random.seed(0)
    weights = np.random.uniform(0,1, (100,))
    values = weights * (1 + np.random.uniform(0,1e-3, (100,)))
    solver = KnapsackSolver(weights, values)
    v, item = solver.greedy(3)
    print(item, solver.total_w(item), solver.total_v(item))

    def summ(idx):
        re = 0
        for i in idx:
            re += values[i]
        return re
    def wei(idx):
        re = 0
        for i in idx:
            re += weights[i]
        return re
    solver2 = SubmodularSolver(list(range(100)), summ)
    out = solver2.solve(wei, 3)
    print(summ(out))
