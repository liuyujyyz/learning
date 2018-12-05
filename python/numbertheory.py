import numpy as np
from time import time 
from decorators import timer

def is_prime(n, plist = None):
    if n == 1:
        return False
    if plist is None:
        plist = range(n)
    for i in plist:
        if i * i > n:
            break
        if i <= 1:
            continue
        if n % i == 0:
            return False
    return True

@timer
def get_all_primes_v2(n):
    # O(n^1.5)
    primes = []
    for i in range(2,n):
        if is_prime(i, primes):
            primes.append(i)
    return primes

@timer
def get_all_primes(n):
    # O(n loglog n)
    free = np.zeros((n,), dtype='uint8')
    primes = []
    free[0] = 1
    free[1] = 1
    idx = 2
    while idx < n:
        if free[idx]:
            idx += 1
            continue
        p = idx
        primes.append(p)
        for i in range(p, n, p):
            free[i] = 1
    return primes

def get_num_of_divisor(n):
    div, power = factory(n)
    re = 1
    for item in power:
        re *= (item + 1)
    return re

@timer
def factory(n):
    assert n >= 2
    pr = int(np.sqrt(n))
    primes = get_all_primes(pr+1)
    pdivisor = []
    power = []
    for item in primes:
        m = 0
        while n % item == 0:
            n = n // item
            m += 1
        if m > 0:
            power.append(m)
            pdivisor.append(item)
    if n > 1:
        pdivisor.append(n)
        power.append(1)
    return (pdivisor, power)

def choose(a, b):
    re = 1
    if a < b:
        return -1
    if a == b:
        return 1
    for i in range(a-b):
        re *= (b + 1 + i)
        re = re // (i + 1)
    return re

if __name__ == '__main__':
    for i in range(2, 33):
        a = factory(2**i-1)
        print(i, a)
