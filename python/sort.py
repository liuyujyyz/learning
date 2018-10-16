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
def qs_p(a):
    return qs(a)

def qs(a, thr = 6):
    L = len(a)
    if L < thr:
        return bubble(a)
    idx = np.random.randint(L)
    prev = []
    post = []
    base = a[idx]
    for i in range(L):
        num = a[i]
        if i == idx:
            continue
        if num < base:
            prev.append(num)
        elif num > base:
            post.append(num)
        else:
            if i < idx:
                prev.append(num)
            else:
                post.append(num)
    if len(prev) > 1:
        prev = qs(prev, thr)
    if len(post) > 1:
        post = qs(post, thr)
    return prev + [base] + post

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
    for l in [100,1000,10000,100000]:
        base = l * np.log(l)
        a = list(np.random.uniform(0, 100, (l,)))
        e = npsort(a)
        b = qs_p(a)
        c = ms(a)
        # d = bubble(a)
