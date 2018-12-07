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
            for i in range(st, n+1, p):
                free[i-(self.maximum+1)] = 1
        idx = 0
        while idx < n-self.maximum:
            if free[idx]:
                idx += 1
                continue
            p = idx + self.maximum + 1
            self.primes.append(p)
            for i in range(idx, n-self.maximum, p):
                free[i] = 1
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

if __name__ == '__main__':
    pass
