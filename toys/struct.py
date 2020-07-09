import numpy as np
from utils import SkillType, BaseAttr, DynamicAttr 


class Skill:
    def __init__(self):
        self.type = SkillType.getRandomType()
        self.attr = DynamicAttr.getRandomType(4)
        self.target = DynamicAttr.getRandomType(3)
        self.ratio = np.random.randint(2,10) / 100.0


class Person:
    def __init__(self):
        self.attrs = {attr.name: np.random.randint(10,100) for attr in BaseAttr}

if __name__ == '__main__':
    a = Person()
    print(a.attrs)


