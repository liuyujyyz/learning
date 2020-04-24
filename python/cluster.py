import numpy as np
import sys, os
import pickle
from tqdm import tqdm
from decorators import timer
#np.random.seed(0)

def random_center(data, k):
    idx = np.random.choice(range(data.shape[0]), k)
    center = data[idx]
    return center

@timer
def nearest(data, center):
    norm1 = (data**2).sum(axis=1, keepdims=True)
    norm2 = (center**2).sum(axis=1, keepdims=True)
    cross = np.dot(data, center.T)
    dist = norm1 + norm2.T - 2 * cross
    idx = np.argmin(dist, axis=1)
    dist = dist[range(dist.shape[0]), idx]
    return idx, dist

def same(array1, array2):
    dist = ((array1-array2)**2).sum()
    return (dist < 0.5)

@timer
def get_center(data, cato, k):
    center = np.zeros((k, data.shape[1]))
    count = np.zeros((k,))
    for i in range(k):
        center[i] = data[cato==i].sum(axis=0)
        s = (cato==i).sum()
        if s > 0:
            center[i] = center[i] / s
    return center

@timer
def Kmeans(data, k):
    center = random_center(data, k)
    cato, dist = nearest(data, center)
    center = get_center(data, cato, k)
    while True:
        cato2, dist = nearest(data, center)
        if same(cato2, cato):
            break
        center = get_center(data, cato2, k)
        cato = cato2
    return cato, dist

if __name__ == '__main__':
    data = np.random.uniform(0,0.01,(1000,100))
    center = np.random.uniform(0, 1, (20, 100))
    for i in range(50):
        data[i*20 : i*20+20] += center
    k = 25 
    bestcato = None
    bestdist = 1e16
    for i in tqdm(range(1000)):
        cato, dist = Kmeans(data, k)
        if dist.sum() < bestdist:
            bestdist = dist.sum()
            bestcato = cato
            print(dist)

