from structure import Physics, Particle
import numpy as np
# np.random.seed(1)
L = 10 
a = np.random.randint(-500, 500, (L, 3))
b = np.random.uniform(10**27, 10**28, (L,))
c = np.random.uniform(-10, 10, (L,))*10**1
v = np.random.uniform(-10,10, (L,3))
v[:,2] = (a[:,0]*v[:,0]+a[:,1]*v[:,1])/(-a[:,2]+1e-8)
plist = [Particle(*a[i], b[i], c[i], v[i]) for i in range(L)]
world = Physics(plist)
world.iterate()


