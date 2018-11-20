import time
import functools
import numpy as np

def repeat(num=0):
    def timer(func):
        def wrapper(*args, **kargs):
            print('test %s'%num)
            return func(*args, **kargs)
        return wrapper
    return timer


def timer(func):
    @functools.wraps(func)
    def fuck(*args, **kargs):
        a = time.time()
        re = func(*args, **kargs)
        b = time.time()
        c = b - a
        print('%s cost time %s' % (func.__name__, c))
        return re
    return fuck

@timer
def add2(L):
    c = sum(L)
    return c

@timer
def add3(L):
    c = np.sum(L)
    return c

@timer
def convert(L):
    return np.array(L)

if __name__ == '__main__':
    for L in range(10):
        A = convert(range(10000*(L+1)))
        c = add2(A)
        c = add3(A)
        print('')
