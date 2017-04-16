import pickle
from monitor import *
from dataprovider import *
from graph import *
input_dims = 28 * 28
hidden_dims = 128
code_len = 32
W = pickle.load(open('AEforMNIST/net.pkl','rb'))
start = W['step']

ins = [input_dims, hidden_dims, code_len, hidden_dims, input_dims]

NB = NetworkBuilder()

for i in range(4):
    NB.add_opr('fc%s'%i, ['a%s'%i], 'z%s'%i, 'FC', {'inp_size': ins[i], 'out_size':ins[i+1], 'W':W['fc%s:W'%(i+1)], 'b':W['fc%s:b'%(i+1)]})
    NB.add_opr('act%s'%i, ['z%s'%i], 'a%s'%(i+1), 'SIGMOID', {})

NB.add_opr('sub1', ['a4', 'a0'], 'diff', 'SUB', {})
NB.add_opr('pow1', ['diff'], 'sq', 'POW', {'k':2})
NB.add_opr('sum1', ['sq'], 'sum', 'SUM', {})
NB.set_loss_var('sum')
NB.set_time_stamp(start)
net = NB.get_desc()
"""
net = {
        'params':['imgs', 'z1','a1','z2','a2','z3','a3','z4','a4','diff','sq','sum'],
        'oprs':{
            'fc1':{'inputs':['imgs'], 'outputs':'z1', 'opr_type':'FC', 'params':{'inp_size':input_dims,'out_size':hidden_dims, 'W':W['fc1:W'], 'b':W['fc1:b']}},
            'act1':{'inputs':['z1'], 'outputs':'a1', 'opr_type':'SIGMOID', 'params':{}},
            'fc2':{'inputs':['a1'], 'outputs':'z2', 'opr_type':'FC', 'params':{'inp_size':hidden_dims, 'out_size':code_len, 'W':W['fc2:W'], 'b':W['fc2:b']}},
            'act2':{'inputs':['z2'], 'outputs':'a2', 'opr_type':'SIGMOID', 'params':{}},
            'fc3':{'inputs':['a2'], 'outputs':'z3', 'opr_type':'FC', 'params':{'inp_size':code_len, 'out_size':hidden_dims, 'W':W['fc3:W'], 'b':W['fc3:b']}},
            'act3':{'inputs':['z3'], 'outputs':'a3', 'opr_type':'SIGMOID', 'params':{}},
            'fc4':{'inputs':['a3'], 'outputs':'z4', 'opr_type':'FC', 'params':{'inp_size':hidden_dims, 'out_size':input_dims, 'W':W['fc4:W'], 'b':W['fc4:b']}},

            'act4':{'inputs':['z4'], 'outputs':'a4', 'opr_type':'SIGMOID', 'params':{}},
            'sub1':{'inputs':['a4', 'imgs'], 'outputs':'diff', 'opr_type':'SUB', 'params':{}},
            'pow1':{'inputs':['diff'], 'outputs':'sq', 'opr_type':'POW', 'params':{'k':2}},
            'sum1':{'inputs':['sq'], 'outputs':'sum', 'opr_type':'SUM', 'params':{}},
        },
        'loss_var': 'sum',
        'time_stamp': start,
    }
"""
G = graph(net)
writer = MonitorWriter('test_graph')
dp = MNIST(with_label = False)

valid_img = dp.valid()
G.forward(data = {'a0':valid_img[:100]})
oup = G.nodes['a4'].value
canvas = np.concatenate([np.concatenate(p, axis = 0) for p in oup.reshape(10, 10 , 28, 28, 1)], axis = 1)
writer.add_image('gen_img', canvas, 0)

canvas = np.concatenate([np.concatenate(p, axis = 0) for p in valid_img[:100].reshape(10, 10 , 28, 28, 1)], axis = 1)
writer.add_image('read_img', canvas, 0)

