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

def make_network():
    def mcp(i,o,k,s,p):
        return {'inp_channel':i,'out_channel':o,'kernel_shape':(k,k),'stride':(s,s),'padding':(p,p)}

    NB = NetworkBuilder()
    NB.add_opr('reshape0', ['img'], 'r0', 'RESHAPE', {'shape':(1,28,28)})
    NB.add_opr('conv0', ['r0'], 'c0', 'CONV', mcp(1,12,4,2,1))
    NB.add_opr('tanh0', ['c0'], 'a0', 'RELU', {})
    NB.add_opr('conv1', ['a0'], 'c1', 'CONV', mcp(12,24,4,2,1))
    NB.add_opr('tanh1', ['c1'], 'a1', 'RELU', {})
    NB.add_opr('reshape1', ['a1'], 'r1', 'RESHAPE', {'shape':(24*7*7,)})
    NB.add_opr('fc0', ['r1'], 'f0', 'FC', {'inp_size':24*49,'out_size':10})
    NB.add_opr('bce', ['f0', 'label'], 'loss', 'BCE', {})
    NB.set_loss_var('loss')
    return NB

def AE():
    NB = NetworkBuilder()
    input_dims = 28 * 28
    hidden_dims = 128
    code_len = 32
    #W = pickle.load(open('AEforMNIST/net.pkl','rb'))
    start = 0#W['step']

    ins = [input_dims, hidden_dims, code_len, hidden_dims, input_dims]

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
    return NB

def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.m:
        writer = MonitorWriter(args.m)
    dp = MNIST(with_label = True)
    
    if not os.path.exists(args.o):
        os.system('mkdir '+args.o)
    
    G = graph(debug = False)
    if args.c:
        G.load(args.c)
        print('load', G.edges)
    else:
        NB = make_network()
        net = NB.get_desc()
        G.construct(net)
    valid_img, valid_label = dp.valid()
    train_img, train_label = next(dp.train(1024))
    for count in range(1000):
        G.forward(data = {'img':train_img, 'label':train_label})
        G.backprop(0.01, weight_decay = 0)
        if args.m:
            writer.add_value('loss', G.loss_var.value, G.step)
        if count % 10 == 0:
            G.save(args.o + '/graph')
            G.forward(data = {'img':valid_img, 'label':valid_label})
            pred = G.nodes['f0'].value
            acc = (np.argmax(pred, axis = 1) == np.argmax(valid_label, axis = 1)).mean()
            print('valid accuracy: %f'%acc)

if __name__ == '__main__':
    main()
