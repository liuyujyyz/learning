import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from game import Player, card
from model import Model

mod = Model()

class NPlayer(Player):
    def __init__(self, ID, base):
        super(Player, self).__init__()

    def action(self):
        if self.count() >= 21:
            return False
        present = self.rep()
        data = Variable(torch.from_numpy(np.array([present]).astype('float32')))
        out = mod(data)
        pred = out.data.numpy()[0]
        if pred[1] > 0.5:
            return True
        return False

def train(max_iter):
    p1 = NPlayer('A', 10)
    optimizer = torch.optim.SGD(mod.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
    for i in range(max_iter):
        card.shuffle()
        p1.revivo()
        p1.draw()
        p1.draw()
        while True:
            act = p1.action()
            if act:
                p1.draw()
            else:
                break
        loss = 21 - p1.count()
        if loss < 0:
            loss = 21
        loss = torch.FloatTensor(int(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    p1 = NPlayer('A',10)
    p2 = Player('B',10)
    count = 0
    for i in range(10000):
        r = new_game(p1, p2)
        if r:
            count += 1
    print(count / 10000)
    train(1000)
    count = 0
    for i in range(10000):
        r = new_game(p1, p2)
        if r:
            count += 1
    print(count / 10000)

