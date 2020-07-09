import numpy as np
from enum import Enum

class RandEnum(Enum):
    @classmethod
    def getRandomType(cls, L=None):
        if L is None:
            return cls(np.random.randint(len(cls)))
        else:
            return cls(np.random.randint(min(L, len(cls))))

class SkillType(RandEnum):
    damage = 0
    buff = 1
    debuff = 2

class BaseAttr(RandEnum):
    base_attack = 0
    base_defence = 1
    base_life = 2
    mat_attack = 3
    mat_defence = 4
    mat_life = 5

class DynamicAttr(RandEnum):
    attack = 0
    defence = 1
    life = 2
    level = 3
    exp = 4
