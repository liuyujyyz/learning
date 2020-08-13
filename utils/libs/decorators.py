import time
import functools
import numpy as np
import os

def timer(func):
    @functools.wraps(func)
    def fuck(*args, **kargs):
        a = time.time()
        re = func(*args, **kargs)
        b = time.time()
        c = b - a
        if os.getenv('DEBUG', 0):
            print('%s cost time %s' % (func.__name__, c))
        return re
    return fuck

