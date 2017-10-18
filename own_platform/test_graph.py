import pickle
from core.monitor import *
from core.dataprovider import *
from core.graph import *
import numpy as np

input_dims = 28 * 28
hidden_dims = 128
code_len = 32
#W = pickle.load(open('AEforMNIST/net.pkl','rb'))
start = 0#W['step']

ins = [input_dims, hidden_dims, code_len, hidden_dims, input_dims]

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
net = NB.get_desc()
G = graph(net, debug = False)
#writer = MonitorWriter('test_graph')
dp = MNIST(with_label = False)

valid_img = dp.valid()
train_img = np.ones((1024, 28*28))
for i in range(100):
    G.forward(data = {'a0':train_img})
    G.backprop(0.001)

G.forward(data = {'a0':np.ones((100,28*28))})
canvas = np.concatenate([np.concatenate(p, axis = 0) for p in oup.reshape(10, 10 , 28, 28, 1)], axis = 1)
writer.add_image('gen_img', canvas, 0)

canvas = np.concatenate([np.concatenate(p, axis = 0) for p in valid_img[:100].reshape(10, 10 , 28, 28, 1)], axis = 1)
writer.add_image('read_img', canvas, 0)

