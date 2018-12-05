import numpy as np
from time import time, sleep
from decorators import timer

def bubble(a):
    for i in range(len(a)):
        for j in range(i+1, len(a)):
            if a[i] > a[j]:
                tmp = a[i]
                a[i] = a[j]
                a[j] = tmp
    return a
            
@timer
def qs_p2(a):
    return qs_fast(a, 0, len(a)-1)

def partition(arr, L, H):
    i = L - 1
    pivot = arr[H]
    for j in range(L, H):
        if arr[j] <= pivot:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i+1], arr[H] = arr[H], arr[i+1]
    return i + 1

def qs_fast(arr, L, H):
    if L < H:
        pi = partition(arr, L, H)
        qs_fast(arr, L, pi-1)
        qs_fast(arr, pi+1, H)

def merge(a, b):
    re = []
    i = 0
    j = 0
    L1 = len(a)
    L2 = len(b)
    while i < L1 and j < L2:
        if a[i] <= b[j]:
            re.append(a[i])
            i += 1
        else:
            re.append(b[j])
            j += 1
    if i < L1:
        re += a[i:]
    if j < L2:
        re += b[j:]
    return re

@timer
def ms(a):
    return merge_sort(a)

def merge_sort(a, thr = 6):
    L = len(a)
    if L < thr:
        return bubble(a)
    if L == 1:
        return a
    l1 = L // 2
    re1 = merge_sort(a[:l1], thr)
    re2 = merge_sort(a[l1:], thr)
    return merge(re1, re2)
   
@timer
def npsort(a):
    b = a.copy()
    b.sort()
    return b

if __name__ == '__main__':
    for l in [100000]:
        a = list(np.random.uniform(0, 100, (l,)))
        e = npsort(a)
        qs_p2(a)
        c = ms(a)
