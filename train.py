import pickle
from oprs import *
from monitor import MonitorWriter
import numpy as np
import argparse
from dataprovider import *
import os

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', help = 'output files directory', default = 'logs')
    parser.add_argument('-c', help = 'file name continued from', default = None)
    parser.add_argument('-m', help = 'monitor name', default = None)
    parser.add_argument('-l', help = 'learning rate', type = float, default = 1e-3)
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()
    
    writer = MonitorWriter(args.m)
    dp = MNIST(with_label = False)
    if not os.path.exists(args.o):
        os.system('mkdir '+args.o)
    input_dims = 28 * 28
    hidden_dims = 128
    code_len = 32
    act = Sigmoid()
    if args.c:
        W = pickle.load(open(args.c,'rb'))
        start = W['step']
        fc1 = FC('fc1', input_dims, hidden_dims,W=W['fc1:W'], b=W['fc1:b'])
        fc2 = FC('fc2', hidden_dims, code_len, W=W['fc2:W'], b=W['fc2:b'])
        fc3 = FC('fc3', code_len, hidden_dims,W=W['fc3:W'],b=W['fc3:b'])
        fc4 = FC('fc4', hidden_dims, input_dims,W=W['fc4:W'],b=W['fc4:b'])
    else:
        start = 0
        fc1 = FC('fc1', input_dims, hidden_dims)
        fc2 = FC('fc2', hidden_dims, code_len)
        fc3 = FC('fc3', code_len, hidden_dims)
        fc4 = FC('fc4', hidden_dims, input_dims)

    def dist(img):
        z1 = fc1(img)
        a1 = act(z1)
        z2 = fc2(a1)
        a2 = act(z2)
        code = a2
        z3 = fc3(a2)
        a3 = act(z3)
        z4 = fc4(a3)
        a4 = act(z4)
        return ((img - a4)**2).sum()/img.shape[0], a4

    bs = 1024
    iter_ = dp.train(bs)
    valid_img = dp.valid()
    lr = args.l
    for j in range(100000):
        i = j + start
        sample_img = next(iter_)

        z1 = fc1(sample_img)
        a1 = act(z1)
        z2 = fc2(a1)
        a2 = act(z2)
        code = a2
        z3 = fc3(a2)
        a3 = act(z3)
        z4 = fc4(a3)
        a4 = act(z4)
        loss = ((a4 - sample_img)**2).sum(axis=1).mean() 
       
        writer.add_value('train_loss', loss, i)

        delta_a4 = 2 * (a4 - sample_img)
        delta_z4 = act.backprop(z4, delta_a4)
        delta_a3 = fc4.backprop(a3, delta_z4, lr)
        delta_z3 = act.backprop(z3, delta_a3)
        delta_a2 = fc3.backprop(a2, delta_z3, lr)
        delta_z2 = act.backprop(z2, delta_a2)
        delta_a1 = fc2.backprop(a1, delta_z2, lr)
        delta_z1 = act.backprop(z1, delta_a1)
        delta_inp = fc1.backprop(sample_img, delta_z1, lr)

        if i % 100 == 0:
            pickle.dump({
                'step': i, 
                'fc1:W':fc1.W, 'fc1:b':fc1.b,
                'fc2:W':fc2.W, 'fc2:b':fc2.b,
                'fc3:W':fc3.W, 'fc3:b':fc3.b,
                'fc4:W':fc4.W, 'fc4:b':fc4.b}, 
                open(args.o + '/net.pkl','wb'))
            acc1, oup = dist(valid_img)
            writer.add_value('valie_loss', acc1, i // 100)
            canvas = np.concatenate([np.concatenate(p, axis = 0) for p in oup[:100].reshape(10, 10 , 28, 28, 1)], axis = 1)
            writer.add_image('gen_img', canvas, i//100)
            canvas = np.concatenate([np.concatenate(p, axis = 0) for p in valid_img[:100].reshape(10, 10 , 28, 28, 1)], axis = 1)
            writer.add_image('read_img', canvas, i//100)
    acc2 = acc(test_img)
    print(acc2)

if __name__ == '__main__':
    main()
