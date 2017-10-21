import torch
import numpy as np
import pickle,json
import cv2

color = ['Spade', 'Heart', 'Club', 'Diamond']
out = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']

class Card():
    def __init__(self):
        self.data = []
        for i in range(4):
            for j in range(13):
                self.data.append([i,j])
        self.data = np.array(self.data)
        self.head = 0

    def shuffle(self):
        pred = np.random.uniform(0,1,(4*13, ))
        idx = np.argsort(pred)
        self.data = self.data[idx]
        self.head = 0

    def next(self):
        re = self.data[self.head]
        self.head += 1
        return re

    def show(self, q):
        col = color[q[0]]
        num = out[q[1]]
        print('%s of %s'%(num, col))
            
card = Card()

class Player():
    def __init__(self, ID, base):
        self.ID = ID
        self.base = base
        self.card = []

    def draw(self):
        self.card.append(card.next())

    def count(self):
        re = 0
        for c in self.card:
            if c[1] < 10:
                re += c[1] + 1
            else:
                re += 10
        return re

    def revivo(self):
        self.card = []

    def action(self):
        if self.count() < 11:
            return True
        return False
    
    def rep(self):
        re = np.zeros((13,))
        for c in self.card:
            re[c[1]] += 1
        return re

    def show(self):
        print('='*10)
        for c in self.card:
            card.show(c)

def new_game(p1,p2):
    card.shuffle()
    p1.revivo()
    p2.revivo()
    p1.draw()
    p2.draw()
    p1.draw()
    p2.draw()
    while True:
        q1 = p1.action()
        if q1:
            p1.draw()
        q2 = p2.action()
        if q2:
            p2.draw()
        if not q1 and not q2:
            break
    h1 = p1.count()
    h2 = p2.count()
    if h1 > 21:
        h1 = 0
    if h2 > 21:
        h2 = 0
    if h1 > h2:
        return True
    else:
        return False

if __name__ == '__main__':
    p1 = Player('A',10)
    p2 = Player('B',10)
    count = 0
    for i in range(100000):
        r = new_game(p1, p2)
        if r:
            count += 1
    print(count / 100000)
