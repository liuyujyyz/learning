import numpy as np
import os
import sys
from tqdm import tqdm
import time
import json
import pickle
from prob import Distribution
from IPython import embed

tianfu = Distribution([0.5, 0.8, 1, 1.2, 1.5, 2, 4], [3, 9, 27, 9, 3, 1, 0.1])

gongfa = Distribution([0.5, 0.8, 1, 1.2, 1.5, 2], [8, 16, 4, 2, 1, 0.5])

diyu = Distribution([0, 0.5, 0.8, 1, 1.2, 1.5, 2], [1, 2, 4, 8, 4, 2, 1])

mijing = Distribution([3, 4, 5], [1, 1, 1])

base = 1

expThr = [base * 365 * (3**i) for i in range(100)]

eventList = [
        None,
        ('tianfu', 'tianfu tisheng', Distribution([1.01 + 0.01 * i for i in range(10)], [2**(10-i) for i in range(10)])),
        ('gongfa', 'gongfa zhuanhuan', gongfa),
        ('diyu', 'diyu zhuanhuan', diyu),
        ('mijing', 'shenmi diyu', mijing), 
        ]
eventDist = Distribution(eventList, [1000, 1, 3, 5, 1])

def get_ratio(tf, gf, dy):
    return (tf ** 3) * np.log(1 + gf) * dy / 4

class Person():
    def __init__(self, ID):
        self.ID = ID
        self.tianfu = tianfu.sample_one() * np.random.uniform(0.9, 1.1)
        self.originTianfu = self.tianfu
        self.exp = 0
        self.gongfa = gongfa.sample_one()
        self.diyu = diyu.sample_one()
        self.lv = 0
        self.log = os.path.join('data', '%s.txt' % ID)
        self.round = 0
        self.diyuCount = 0
        if max(self.tianfu, self.gongfa, self.diyu) > 1.95:
            print(ID)
        self.attr = {
                'ID': self.ID,
                'tianfu': self.tianfu,
                'gongfa': self.gongfa,
                'diyu': self.diyu,
                }
        with open(self.log, 'a') as fout:
            fout.write(json.dumps(self.attr) + '\n')

    def update(self, event):
        self.round += 1
        outstr = ''
        delta = base * (1.2**self.lv) * get_ratio(self.tianfu, self.gongfa, self.diyu)
        self.exp += delta
        if self.tianfu >= 2 and self.diyu >= 3 and self.gongfa >= 1.5:
            if np.random.uniform(0,1) < 0.2:
                self.gongfa *= 1.1
        if self.exp > expThr[self.lv]:
            self.lv += 1
            self.gongfa *= 0.95
            outstr += 'level up to %s\n' % self.lv
        if event is not None:
            typ, discription, dist = event
            if typ == 'tianfu':
                multi = dist.sample_one()
                newtianfu = self.tianfu * multi
                outstr += discription + ', from %.4f to %.4f\n' % (self.tianfu, newtianfu)
                self.tianfu = newtianfu
            if typ == 'gongfa':
                newgongfa = dist.sample_one()
                if newgongfa > self.gongfa:
                    outstr += discription + ', from %.2f to %.2f\n' % (self.gongfa, newgongfa)
                    self.gongfa = newgongfa
#                else:
#                    outstr += discription + ', %.2f too bad, discard\n' % newgongfa
            if typ in ['diyu', 'mijing']:
                newdiyu = dist.sample_one()
                if newdiyu < self.diyu and self.diyuCount < 10:
                    self.diyuCount += 1
                else:
                    outstr += discription + ', from %.2f to %.2f\n' % (self.diyu, newdiyu)
                    self.diyu = newdiyu
                    self.diyuCount = 0

        if outstr != '':
            with open(self.log, 'a') as fout:
                fout.write('year %.2f:\n'%(self.round * 10/ 365))
                fout.write(outstr)


def main():
    L = 200
    perList = [Person(i) for i in range(L)]
    personEventList = [[] for i in range(L)]
    for i in tqdm(range(50000)):
        for j in range(L):
            item = perList[j]
            if len(personEventList[j]) > 0:
                event = personEventList[j].pop(0)
                item.update(event)
            else:
                event = eventDist.sample_one()
                if event is not None and event[0] == 'mijing':
                    personEventList[j] += [eventList[3]] * 10
                item.update(event)
    lvList = [item.lv for item in perList]
    maxLv = max(lvList)
    minLv = min(lvList)
    print(maxLv, minLv)

if __name__ == '__main__':
    main()

        

