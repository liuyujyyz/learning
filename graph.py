from oprs import *
#oprs dict name:[inputs, outputs, type, params]
#params list name

class node():
    def __init__(self, name):
        self.name = name
        self.inputs = []
        self.outputs = []
        self.shape = None
        self.value = None
        self.delta = 0

    def show(self):
        print(self.name, self.inputs, self.outputs, self.value)

class edge():
    def __init__(self, name, inputs, outputs, opr_type, params):
        func = func_dict[opr_type]
        self.func = func(name, **params)
        self.name = name
        self.input = inputs
        self.output = outputs

    def valid(self, nodes):
        for f in self.input:
            if nodes[f].value is None:
                return False
        return True

    def fw(self, nodes):
        q = [nodes[f].value for f in self.input]
        return self.func(q)

    def bp(self, nodes, lr):
        q = [nodes[f].value for f in self.input]
        L = self.func.backprop(q, nodes[self.output].delta, lr)
        for i in range(len(self.input)):
            nodes[self.input[i]].delta += L[i]


class graph():
    def __init__(self, net_graph):
        self.nodes = {}
        for key in net_graph['params']:
            self.nodes[key] = node(key)
        self.edges = {}
        for key in net_graph['oprs']:
            data = net_graph['oprs'][key]
            self.edges[key] = edge(key, **data)
            for p in data['inputs']:
                self.nodes[p].outputs.append(key)
            self.nodes[data['outputs']].inputs.append(key)
        self.loss_var = self.nodes[net_graph['loss_var']]
        for u in self.nodes:
            self.nodes[u].show()

    def forward(self, data):
        M = list(self.edges.keys())
        for key in self.nodes:
            self.nodes[key].value = None
        for key in data:
            self.nodes[key].value = data[key]

        while M:
            for opr in M:
                if self.edges[opr].valid(self.nodes):
                    val = self.edges[opr].fw(self.nodes)
                    self.nodes[self.edges[opr].output].value = val
                    M.remove(opr)

        print(self.loss_var.value)

    def backprop(self):
        #Not Implemented
        pass

class NetworkBuilder():
    def __init__(self, name = 'HAHA'):
        self.name = name
        self.params = []
        self.oprs = {}
        self.time_stamp = 0
        self.loss_var = None

    def add_opr(self, name, inputs, outputs, opr_type, params):
        assert not (name in self.oprs), 'duplicated name'
        self.oprs[name] = {
                'inputs': inputs,
                'outputs': outputs,
                'opr_type': opr_type,
                'params': params,
                }
        for p in inputs:
            if not (p in self.params):
                self.params.append(p)
        if not(outputs in self.params):
            self.params.append(outputs)

    def set_loss_var(self, loss):
        self.loss_var = loss

    def set_time_stamp(self, ts):
        self.time_stamp = ts

    def get_desc(self):
        return {'params':self.params, 'oprs':self.oprs, 'loss_var':self.loss_var, 'time_stamp':self.time_stamp}
