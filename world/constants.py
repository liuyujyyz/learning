import numpy as np
from utils import same

class PhyVector:
    dimConst = ['m', 'kg', 's', 'A', 'K', 'mol', 'cd']
    def __init__(self, phyvalues, dimension=None):
        if dimension is not None:
            self.value = np.array(phyvalues)
            self.dimension = np.array(dimension)
        else:
            x, y, z = phyvalues
            assert same(x.dimension, y.dimension) and same(x.dimension, z.dimension)
            self.value = np.array([x.value, y.value, z.value])
            self.dimension = x.dimension

    def __add__(self, addend):
        assert isinstance(addend, PhyVector) and same(self.dimension, addend.dimension), (type(addend), self.dimension, addend.dimension)
        addvalue = self.value + addend.value
        dimension = self.dimension
        return PhyVector(addvalue, dimension)

    def __radd__(self, addend):
        assert isinstance(addend, PhyVector) and same(self.dimension, addend.dimension)
        addvalue = self.value + addend.value
        dimension = self.dimension
        return PhyVector(addvalue, dimension)

    def __sub__(self, rhs):
        assert isinstance(rhs, PhyVector) and same(self.dimension, rhs.dimension)
        subvalue = self.value - rhs.value
        dimension = self.dimension
        return PhyVector(subvalue, dimension)

    def __rsub__(self, lhs):
        assert isinstance(lhs, PhyVector) and same(self.dimension, lhs.dimension)
        subvalue = lhs.value - self.value
        dimension = self.dimension
        return PhyVector(subvalue, dimension)

    def __mul__(self, multiplier):
        if not isinstance(multiplier, PhyValue):
            multiplier = PhyValue(multiplier, [0]*7)
        mulvalue = self.value * multiplier.value
        dimension = self.dimension + multiplier.dimension
        return PhyVector(mulvalue, dimension)

    def __rmul__(self, multiplier):
        if not isinstance(multiplier, PhyValue):
            multiplier = PhyValue(multiplier, [0]*7)
        mulvalue = self.value * multiplier.value
        dimension = self.dimension + multiplier.dimension
        return PhyVector(mulvalue, dimension)

    def orient(self):
        r = self.value / (self.value**2).sum()**0.5
        return PhyVector(r, [0]*7)

    def norm(self):
        r = (self.value**2).sum()**0.5
        return PhyValue(r, self.dimension)

    def normsqr(self):
        r = (self.value**2).sum()
        return PhyValue(r, self.dimension*2)

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



class PhyValue:
    dimConst = ['m', 'kg', 's', 'A', 'K', 'mol', 'cd']
    def __init__(self, value, dimension):
        self.value = value
        if isinstance(dimension, str):
            self.dimension = np.array([int(dimension==self.dimConst[i]) for i in range(7)])
        else:
            self.dimension = np.array(dimension)

    def __add__(self, addend):
        if not isinstance(addend, PhyValue):
            addend = PhyValue(addend, [0]*7)
        assert same(self.dimension, addend.dimension)
        addvalue = self.value + addend.value
        dimension = self.dimension
        return PhyValue(addvalue, dimension)

    def __radd__(self, addend):
        if not isinstance(addend, PhyValue):
            addend = PhyValue(addend, [0]*7)
        assert same(self.dimension, addend.dimension)
        addvalue = self.value + addend.value
        dimension = self.dimension
        return PhyValue(addvalue, dimension)

    def __sub__(self, rhs):
        if not isinstance(rhs, PhyValue):
            rhs = PhyValue(rhs, [0]*7)
        assert same(self.dimension, rhs.dimension)
        subvalue = self.value - rhs.value
        dimension = self.dimension
        return PhyValue(subvalue, dimension)

    def __rsub__(self, lhs):
        if not isinstance(lhs, PhyValue):
            lhs = PhyValue(lhs, [0]*7)
        assert same(self.dimension, lhs.dimension), (self.dimension, lhs.dimension)
        subvalue = lhs.value - self.value
        dimension = self.dimension
        return PhyValue(subvalue, dimension)

    def __mul__(self, multiplier):
        if not isinstance(multiplier, PhyValue) and not isinstance(multiplier, PhyVector):
            multiplier = PhyValue(multiplier, [0]*7)
        mulvalue = self.value * multiplier.value
        dimension = self.dimension + multiplier.dimension
        if isinstance(multiplier, PhyVector):
            return PhyVector(mulvalue, dimension)
        return PhyValue(mulvalue, dimension)

    def __rmul__(self, multiplier):
        if not isinstance(multiplier, PhyValue) and not isinstance(multiplier, PhyVector):
            multiplier = PhyValue(multiplier, [0]*7)
        mulvalue = self.value * multiplier.value
        dimension = self.dimension + multiplier.dimension
        if isinstance(multiplier, PhyVector):
            return PhyVector(mulvalue, dimension)
        return PhyValue(mulvalue, dimension)

    def __truediv__(self, rhs):
        if not isinstance(rhs, PhyValue):
            rhs = PhyValue(rhs, [0]*7)
        divvalue = self.value / rhs.value
        dimension = self.dimension - rhs.dimension
        return PhyValue(divvalue, dimension)

    def __rtruediv__(self, lhs):
        if not isinstance(lhs, PhyValue) and not isinstance(lhs, PhyVector):
            lhs = PhyValue(lhs, [0]*7)
        divvalue = lhs.value / self.value
        dimension = lhs.dimension - self.dimension
        if isinstance(lhs, PhyVector):
            return PhyVector(divvalue, dimension)
        return PhyValue(divvalue, dimension)
    
    def __pow__(self, rhs):
        powvalue = self.value ** rhs
        dimension = (self.dimension * rhs).astype('int')
        return PhyValue(powvalue, dimension)

    def __neg__(self):
        return PhyValue(-self.value, self.dimension)

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
        re = re + vp
        if vn != '':
            re = re + '/' + vn
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


if __name__ == '__main__':
    m1 = PhyValue(1, 'kg')
    m2 = PhyValue(1, 'kg')
    e1 = 1e-7 * A * s
    e2 = 1e-7 * A * s
    e0 = 1e9*Conste
    r = PhyValue(1, 'm')
    F1 = ConstG*m1*m2/(r**2)
    F2 = ConstK*(e1-e0)*e2/(r*r)
    L = F1 - F2
    print(L)

