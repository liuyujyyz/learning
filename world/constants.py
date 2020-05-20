import numpy as np
from utils import same


class PhyUnit:
    dimConst = ['m', 'kg', 's', 'A', 'K', 'mol', 'cd']
    def __init__(self, value, dimension):
        self.value = np.array(value)
        self.dimension = np.array(dimension)

    def __add__(self, addend):
        assert isinstance(addend, PhyUnit) and same(self.dimension, addend.dimension), (type(addend), self.dimension, addend.dimension)
        addvalue = self.value + addend.value
        dimension = self.dimension
        return type(self)(addvalue, dimension)

    def __radd__(self, addend):
        assert isinstance(addend, PhyUnit) and same(self.dimension, addend.dimension)
        addvalue = self.value + addend.value
        dimension = self.dimension
        return type(self)(addvalue, dimension)

    def __sub__(self, rhs):
        assert isinstance(rhs, PhyUnit) and same(self.dimension, rhs.dimension)
        subvalue = self.value - rhs.value
        dimension = self.dimension
        return type(self)(subvalue, dimension)

    def __rsub__(self, lhs):
        assert isinstance(lhs, PhyUnit) and same(self.dimension, lhs.dimension)
        subvalue = lhs.value - self.value
        dimension = self.dimension
        return type(self)(subvalue, dimension)

    def __mul__(self, multiplier):
        if not isinstance(multiplier, PhyUnit):
            multiplier = PhyUnit(multiplier, [0]*7)
        mulvalue = self.value * multiplier.value
        dimension = self.dimension + multiplier.dimension
        return type(self)(mulvalue, dimension)

    def __rmul__(self, multiplier):
        if not isinstance(multiplier, PhyUnit):
            multiplier = PhyUnit(multiplier, [0]*7)
        mulvalue = self.value * multiplier.value
        dimension = self.dimension + multiplier.dimension
        return type(self)(mulvalue, dimension)

    def __truediv__(self, rhs):
        if not isinstance(rhs, PhyUnit):
            rhs = PhyUnit(rhs, [0]*7)
        divvalue = self.value / rhs.value
        dimension = self.dimension - rhs.dimension
        return type(self)(divvalue, dimension)

    def __rtruediv__(self, lhs):
        if not isinstance(lhs, PhyUnit):
            lhs = PhyValue(lhs, [0]*7)
        divvalue = lhs.value / self.value
        dimension = lhs.dimension - self.dimension
        return type(lhs)(divvalue, dimension)
    
    def __pow__(self, rhs):
        powvalue = self.value ** rhs
        dimension = (self.dimension * rhs).astype('int')
        return type(self)(powvalue, dimension)

    def __neg__(self):
        return type(self)(-self.value, self.dimension)
 
    def orient(self):
        r = self.value / (self.value**2).sum()**0.5
        return type(self)(r, [0]*7)

    def norm(self):
        r = (self.value**2).sum()**0.5
        return type(self)(r, self.dimension)

    def normsqr(self):
        r = (self.value**2).sum()
        return type(self)(r, self.dimension*2)


class PhyVector(PhyUnit):
    def __init__(self, phyvalues, dimension=None):
        if dimension is not None:
            super().__init__(phyvalues, dimension)
        else:
            x, y, z = phyvalues
            assert same(x.dimension, y.dimension) and same(x.dimension, z.dimension)
            super().__init__([x.value, y.value, z.value], x.dimension)

    def __or__(self, rhs):
        l, m, n = self.value 
        o, p, q = rhs.value
        value = np.array([m*q-n*p, n*o-l*q, l*p-m*o])
        dimension = self.dimension + rhs.dimension
        return PhyVector(value, dimension)

    def __str__(self):
        re = '[%.6g, %.6g, %.6g]'%(self.value[0], self.value[1], self.value[2])
        vp = ''
        vn = ''
        for i in range(7):
            if self.dimension[i] > 0:
                vp += self.dimConst[i]
                if self.dimension[i] != 1:
                    vp += str(self.dimension[i])
            elif self.dimension[i] < 0:
                vn += self.dimConst[i] 
                if self.dimension[i] != -1:
                    vn += str(-self.dimension[i])
        re = re + vp
        if vn != '':
            re = re + '/' + vn
        return re


class PhyValue(PhyUnit):
    def __init__(self, value, dimension):
        if isinstance(dimension, str):
            dimension = np.array([int(dimension==self.dimConst[i]) for i in range(7)])
        super().__init__(value, dimension)

    def __str__(self):
        re = '%.6g'%self.value
        vp = ''
        vn = ''
        for i in range(7):
            if self.dimension[i] > 0:
                vp += self.dimConst[i]
                if self.dimension[i] != 1:
                    vp += str(self.dimension[i])
            elif self.dimension[i] < 0:
                vn += self.dimConst[i] 
                if self.dimension[i] != -1:
                    vn += str(-self.dimension[i])
        re = re + '('+ vp
        if vn != '':
            re = re + '/' + vn
        re += ')'
        return re


# dimension = [m, kg, s, A, K, mol, cd]
pi = np.pi
# Coulomb Constant
ConstK = PhyValue(8987551787.3681764, [3, 1, -4, -2, 0, 0, 0])
# Gravitational Constant
ConstG = PhyValue(6.67259e-11, [3, -1, -2, 0, 0, 0, 0])
# Avogadro Constant
ConstN_A = PhyValue(6.02214076e23, [0, 0, 0, 0, 0, -1, 0])
# Light Speed
Constc = PhyValue(299792458, [1, 0, -1, 0, 0, 0, 0])
# Electronic Charge
Conste = PhyValue(1.602176487e-19, [0, 0, 1, 1, 0, 0, 0])
# Plank Constant
Consth = PhyValue(6.62606896e-34, [2, 1, -1, 0, 0, 0, 0])

m = PhyValue(1, 'm')
s = PhyValue(1, 's')
kg = PhyValue(1, 'kg')
K = PhyValue(1, 'K')
A = PhyValue(1, 'A')
mol = PhyValue(1, 'mol')
cd = PhyValue(1, 'cd')

N = kg * m / (s*s)
C = A * s
J = N * m
V = J / C

if __name__ == '__main__':
    m1 = kg 
    m2 = kg
    e1 = 1e-7 * C
    e2 = 1e-7 * C
    e0 = 1e9 * Conste
    r = m
    F1 = ConstG*m1*m2/(r**2)
    F11 = PhyVector([F1, F1, F1])
    F2 = ConstK*(e1-e0)*e2/(r*r)
    F22 = PhyVector([F2, F2, -F2])
    L = F1 - F2
    M = F11 | F22
    print(M)

