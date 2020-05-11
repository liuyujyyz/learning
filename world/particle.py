from constants import *
from IPython import embed

N = kg * m / (s*s)

def force(p1, p2):
    direction = p2.position - p1.position
    r = direction.normsqr() 
    F1 = ConstG*p1.mass*p2.mass/r
    F2 = -ConstK*p1.charge*p2.charge/r
    F = F1 + F2
    outF = direction.orient()*F 
    return outF

def forceUnion(p, pset):
    f = PhyVector([0*N, 0*N, 0*N])
    for item in pset:
        tmpf = force(p, item)
        f += tmpf
    return f 

class Particle:
    def __init__(self, position, velocity, mass, charge):
        x, y, z = position
        self.position = PhyVector([x*m, y*m, z*m])
        vx, vy, vz = velocity
        self.velocity = PhyVector([vx*m/s, vy*m/s, vz*m/s])
        self.mass = mass * kg
        self.charge = charge * Conste

    def _mass(self):
        vsqr = self.velocity.normsqr()
        mass = self.mass / (1-vsqr/(Constc**2))**0.5
        return mass

    def move(self, f, t):
        if not isinstance(t, PhyValue):
            t = t * s
        a = f / self.mass
        self.position += self.velocity * t
        self.velocity += a * t
        
    def __str__(self):
        re = 'mass: ' + str(self.mass) + ' charge: ' + str(self.charge)
        return re
