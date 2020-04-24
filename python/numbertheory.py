import numpy as np
from time import time 
from decorators import timer

class Numbers:
    def __init__(self):
        self.primes = [2]
        self.maximum = 2

    def is_prime(self, n):
        self.extend(n)
        return (n in self.primes)

    def extend(self, n):
        if n <= self.maximum:
            return
        free = np.zeros((n-self.maximum,), dtype='uint8') # 0 - max+1, n-max-1 - n
        for p in self.primes:
            st = (self.maximum // p + 1) * p
            free[st-self.maximum-1:n-self.maximum:p] = 1
        idx = 0
        while idx < n-self.maximum:
            if free[idx]:
                idx += 1
                continue
            p = idx + self.maximum + 1
            self.primes.append(p)
            free[idx:n-self.maximum:p] = 1
        self.maximum = n
        return

    def get_num_of_divisor(self, n):
        div, power = self.factory(n)
        re = 1
        for item in power:
            re *= (item + 1)
        return re

    def factory(self, n):
        if n < 2:
            return None
        pr = int(np.sqrt(n))
        if pr + 1 > self.maximum:
            self.extend(pr + 1)
        pdivisor = []
        power = []
        for item in self.primes:
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

@timer 
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

def count2(n, x):
    c = 0
    base = str(x)
    for item in range(n):
        if item % 2 == 1:
            continue
        s = str(item + 1)
        for i in s:
            if i == base:
                c += 1
    return c

@timer
def count3(n, x):
    if n < 100:
        return count2(n, x)
    strnum = str(n)
    L = len(strnum)
    head = int(strnum[0])
    tail = int(strnum[1:])
    tmp0 = count3(tail, x) + (tail + 1) // 2 * (head == x)
    tmp1 = head * count3(10**(L-1) - 1, x)
    if x < head:
        out = 10**(L-1) // 2
    else:
        out = 0
    return out + tmp0 + tmp1


if __name__ == '__main__':
    print(count3(2147483647, 3))
