import numpy as np
import sys, os
import pickle
from tqdm import tqdm

np.random.seed(0)

def random_center(data, k):
    idx = np.random.choice(range(data.shape[0]), k)
    center = data[idx]
    return center

def nearest(data, center):
    norm1 = (data**2).sum(axis=1, keepdims=True)
    norm2 = (center**2).sum(axis=1, keepdims=True)
    cross = np.dot(data, center.T)
    dist = norm1 + norm2.T - 2 * cross
    idx = np.argmin(dist, axis=1)
    dist = dist[range(dist.shape[0]), idx].sum()
    return idx, dist

def same(array1, array2):
    dist = ((array1-array2)**2).sum()
    return (dist < 0.5)

def get_center(data, cato):
    center = np.zeros((k, data.shape[1]))
    count = np.zeros((k,))
    for i in range(len(data)):
        count[cato[i]] += 1
        center[cato[i]] += data[i]
    for i in range(k):
        if count[i] > 0:
            center[i] /= count[i]
    return center

def Kmeans(data, k):
    center = random_center(data, k)
    cato, dist = nearest(data, center)
    center = get_center(data, cato)
    while True:
        cato2, dist = nearest(data, center)
        if same(cato2, cato):
            break
        center = get_center(data, cato2)
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
        if dist < bestdist:
            bestdist = dist
            bestcato = cato
            print(dist)

