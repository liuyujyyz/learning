import json
import os
from IPython import embed
from tqdm import tqdm
import numpy as np

def analyzer(filename):
    bigEvent = {
            'ID': None,
            'mijing': [],
            'lvs': [],
            }
    with open(filename, 'r') as fin:
        data = fin.readlines()
        attr = json.loads(data[0])
        bigEvent['ID'] = attr['ID']
        L = len(data) - 1
        level = 0
        i = 1
        year = 0
        while i < len(data):
            raw = data[i]
            if raw.startswith('year'):
                year = int(float(raw[:-2].split(' ')[-1]) * 100)
                i += 1
                continue
            if raw.startswith('shenmi'):
                mijing_level = raw.strip().split(' ')[-1]
                mijing_level = int(float(mijing_level))
                bigEvent['mijing'].append([year, mijing_level, level])
                i += 1
                continue
            if raw.startswith('level'):
                level = int(raw.strip().split(' ')[-1])
                i += 1
                bigEvent['lvs'].append(year)
                continue
            i += 1
    bigEvent['lvs'] += [1e6] * (15 - len(bigEvent['lvs']))
    return bigEvent


def getEventList(directory):
    out = []
    L = os.walk(directory)
    p, d, f = next(L)
    for item in f:
        filename = os.path.join(p, item)
        event = analyzer(filename)
        event['ID'] = directory + '.' + str(event['ID'])
        out.append(event)
    return out

class Mijing():
    def __init__(self, mtime, level, ID):
        self.start = mtime
        self.end = mtime
        self.level = level
        self.attendence = [ID]

    def overlap(self, mtime, level, ID):
        if level != self.level:
            return False
        if mtime < self.start - 5 or mtime > self.end + 5:
            return False
        self.start = min(self.start, mtime)
        self.end = max(self.end, mtime)
        self.attendence.append(ID)
        return True

def main():
    name = 'data'
    L = getEventList(name)
    analyze_lv(L)

def analyze_lv(L):
    for i in range(15):
        H = [item['lvs'][i] for item in L]
        idx = np.argmin(H)
        print(i, L[idx]['ID'])

def analyze_mijing(L):
    mijingList = []
    for item in tqdm(L):
        ID = item['ID']
        flag = False
        for mj in item['mijing']:
            mtime, level, plv = mj
            for mj_item in mijingList:
                re = mj_item.overlap(mtime, level, (ID, plv))
                if re:
                    flag = True
                    break
        if not flag:
            mijingList.append(Mijing(mtime, level, (ID, plv)))
    maxLen = 0
    maxItem = None
    for item in mijingList:
        if len(item.attendence) > maxLen:
            maxLen = len(item.attendence)
            maxItem = item
    print(maxItem.attendence)
    embed()

if __name__ == '__main__':
    main()
