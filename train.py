import pickle
from core.monitor import *
from core.dataprovider import *
from core.graph import *
import numpy as np
import argparse
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
    
    #writer = MonitorWriter(args.m)
    dp = MNIST(with_label = False)
    if not os.path.exists(args.o):
        os.system('mkdir '+args.o)
    input_dims = 28 * 28
    hidden_dims = 128
    code_len = 32
    #W = pickle.load(open('AEforMNIST/net.pkl','rb'))
    start = 0#W['step']

    ins = [input_dims, hidden_dims, code_len, hidden_dims, input_dims]
    G = graph(debug = False)
    if args.c:
        G.load(args.c)
    else:
        NB = NetworkBuilder()

        for i in range(4):
        #    NB.add_opr('fc%s'%i, ['a%s'%i], 'z%s'%i, 'FC', {'inp_size': ins[i], 'out_size':ins[i+1], 'W':W['fc%s:W'%(i+1)], 'b':W['fc%s:b'%(i+1)]})
            NB.add_opr('fc%s'%i, ['a%s'%i], 'z%s'%i, 'FC', {'inp_size': ins[i], 'out_size':ins[i+1]})
            NB.add_opr('act%s'%i, ['z%s'%i], 'a%s'%(i+1), 'SIGMOID', {})

        NB.add_opr('reshape0', ['a4'], 'r0', 'RESHAPE', {'shape':(1,28,28)})
        NB.add_opr('conv', ['r0'], 'c0', 'CONV', {'inp_channel':1,'out_channel':1,'kernel_shape':(3,3),'stride':(1,1),'padding':(1,1)})
        NB.add_opr('reshape1', ['c0'], 'r1', 'RESHAPE', {'shape':(28*28,)})

        NB.add_opr('sub1', ['r1', 'a0'], 'diff', 'SUB', {})
        NB.add_opr('pow1', ['diff'], 'sq', 'POW', {'k':2})
        NB.add_opr('sum1', ['sq'], 'sum', 'SUM', {})
        NB.set_loss_var('sum')
        NB.set_time_stamp(start)
        NB.save(args.o+'/net')
        net = NB.get_desc()
        G.construct(net)
    #writer = MonitorWriter('test_graph')
    dp = MNIST(with_label = False)

    valid_img = dp.valid()
    train_img = next(dp.train(1024))
    L = 100000
    count = 0
    while L > 50:
        G.forward(data = {'a0':train_img})
        G.backprop(0.001, weight_decay = 1e-4)
        L = G.loss_var.value/1024
        if count % 10 == 0:
            G.save(args.o + '/graph')
            oup = G.nodes['r1'].value
            canvas = np.concatenate([np.concatenate(p, axis = 0) for p in oup[:100].reshape(10, 10 , 28, 28, 1)], axis = 1) * 256
            cv2.imwrite('test_img.jpg', canvas.astype('uint8'))
        count += 1
if __name__ == '__main__':
    main()
