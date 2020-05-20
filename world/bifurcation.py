import numpy as np
from matplotlib import pyplot as plt
import os

def plotBifurDiag(st, ed, func, name): 
    if not os.path.isdir('data/%s'%name):
        os.mkdir('data/%s'%name)

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 9), sharex=True)

    n = 10000
    r = np.linspace(st, ed, n)
    iterrations = 1000
    last = 100
    x = 1e-5*(np.ones(n))

    for i in range(iterrations):
        x = func(r, x)
        if i >= iterrations - last:
            ax1.plot(r, x, ',k', alpha=.25)
    ax1.set_xlim(st, ed)

    plt.savefig('data/%s/%s_%s.png'%(name, st, ed))


if __name__ == '__main__':
    f = lambda r, x: r*x/(1+np.sin(x))
    for st in [-1]:
        plotBifurDiag(st, 4, f, 'cubiclogistic')
