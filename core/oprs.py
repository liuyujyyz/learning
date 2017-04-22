import numpy as np
import cv2
class MultiOprs():
    def __init__(self, name):
        self.name = name

class Sum(MultiOprs):
    def __call__(self, inputs):
        A = inputs[0]
        return A.sum()

    def backprop(self, inputs, delta, lr,weight_decay):
        A = inputs[0]
        return [np.ones(A.shape)]

class Add(MultiOprs):
    def __call__(self, inputs):
        A, B = inputs
        return A + B

    def backprop(self, inputs, delta, lr,weight_decay):
        return [delta, delta]

class Sub(MultiOprs):
    def __call__(self, inputs):
        A, B = inputs
        return A - B

    def backprop(self, inputs, delta, lr,weight_decay):
        return [delta, -delta]

class Mul(MultiOprs):
    def __call__(self, inputs):
        A, B = inputs
        return A * B

    def backprop(self, inputs, delta, lr,weight_decay):
        A, B = inputs
        return [delta * B, delta * A]

class Div(MultiOprs):
    def __call__(self, inputs):
        A, B = inputs
        return A / B

    def backprop(self, inputs, delta, lr,weight_decay):
        A, B = inputs
        return [delta / B, - delta * A / (B**2)]

class Exp(MultiOprs):
    def __call__(self, inputs):
        _ = inputs[0]
        return np.exp(_)

    def backprop(self, inputs, delta, lr,weight_decay):
        _ = inputs[0]
        return [delta * np.exp(_)]

class Reshape(MultiOprs):
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape

    def total(self, shape):
        re = 1
        for i in shape:
            re *= i
        return re

    def __call__(self, inputs):
        if self.total(inputs[0].shape) == self.total(self.shape):
            return inputs[0].reshape(self.shape)
        elif self.total(inputs[0].shape[1:]) == self.total(self.shape):
            return inputs[0].reshape((inputs[0].shape[0],)+self.shape)
        else:
            assert False, 'shape not match'

    def backprop(self, inputs, delta, lr,weight_decay):
        return [delta.reshape(inputs[0].shape)]

class Pow():
    def __init__(self, name, k):
        self.name = name
        self.k = k
        pass

    def __call__(self, inputs):
        _ = inputs[0]
        return _ ** self.k

    def backprop(self, inputs, delta, lr,weight_decay):
        _ = inputs[0]
        return [self.k * delta * (_ ** (self.k - 1))]

class Concat():
    def __init__(self, name, axis):
        self.axis = axis
        self.name = name
        pass

    def __call__(self, inputs):
        return np.concatenate(inputs, axis = self.axis)

    def backprop(self, inputs, delta, lr,weight_decay):
        re = []
        shift = np.moveaxis(delta, self.axis, 0)
        start = 0
        for i in inputs:
            l = i.shape[self.axis]
            re.append(np.moveaxis(delta[start:start+l], 0, self.axis))
            start = start + l
        return re

class ReLU(MultiOprs):
    def __call__(self, inputs):
        re = inputs[0]
        re[re < 0] = 0
        return re

    def backprop(self, inputs, delta, lr,weight_decay):
        _ = inputs[0]
        re = delta * ((_ > 0).astype('int'))
        return [re]

class Sigmoid(MultiOprs):
    def __call__(self, inputs):
        e = np.exp(inputs[0])
        return 1 - 1.0/(1+e)

    def backprop(self, inputs, delta, lr,weight_decay):
        e = np.exp(inputs[0])
        sig = 1 - 1.0/(1+e)
        return [sig*(1-sig)*delta]

class Tanh(MultiOprs):
    def __call__(self, inputs):
        return np.tanh(inputs[0])

    def backprop(self, inputs, delta, lr,weight_decay):
        return [(1-np.tanh(inputs[0])**2)*delta]


class FC():
    def __init__(self, name, inp_size, out_size, W = None, b = None):
        self.name = name
        self.inp_size = inp_size
        self.out_size = out_size
        if W is None:
            self.W = np.random.uniform(-1,1,(inp_size, out_size))/np.sqrt(inp_size)
        else:
            self.W = W
        if b is None:
            self.b = np.zeros((1, out_size))
        else:
            self.b = b
        assert self.W.shape[0] == inp_size and self.W.shape[1] == out_size; 'not matching'

    def __call__(self, inputs):
        return inputs[0]@self.W + self.b

    def backprop(self, inputs, delta, lr,weight_decay):
        grad_weights = inputs[0].T @ delta / inputs[0].shape[0]
        grad_bias = delta.mean(axis = 0, keepdims = True)
        inp_delta = delta @ self.W.T
        grad_weights += weight_decay * self.W
        grad_bias += weight_decay * self.b
        self.W -= lr * grad_weights
        self.b -= lr * grad_bias
        return [inp_delta]

class UpSampling():
    def __init__(self, name, scale = (1,1)):
        self.name = name
        self.scale = scale

    def __call__(self, inputs):
        N,C,H,W = inputs[0].shape
        NH = int(self.scale[0] * H)
        NW = int(self.scale[1] * W)
        img = np.zeros((N,C,H+1,W+1))
        img[:,:,:-1,:-1] = inputs[0]
        out = np.zeros((N,C,NH,NW))
        for i in range(NH):
            for j in range(NW):
                pi = int(i / self.scale[0])# i = pi * l + (pi+1)*(1-l) = (pi+1) - l
                pj = int(j / self.scale[1])
                li = pi + 1 - i
                lj = pj + 1 - j
                out[:,:,i,j] = (img[:,:,pi,pj] * li + img[:,:,pi+1,pj]*(1-li))*lj + (img[:,:,pi,pj+1]*li + img[:,:,pi+1,pj+1]*(1-li))*(1-lj)
        return out

    def backprop(self, inputs, delta, lr, weight_decay):
        img = inputs[0]
        N,C,H,W = img.shape
        NH, NW = delta.shape[2:]
        delta_in = np.zeros((N,C,H+1,W+1))
        for i in range(NH):
            for j in range(NW):
                pi = int(i/self.scale[0])
                pj = int(j/self.scale[1])
                li = pi+1-i
                lj = pj+1-j
                delta_in[:,:,pi,pj] += delta[:,:,i,j] * li * lj
                delta_in[:,:,pi+1,pj] += delta[:,:,i,j] * (1 - li) * lj
                delta_in[:,:,pi, pj+1] += delta[:,:,i,j] * li*(1-lj)
                delta_in[:,:,pi+1,pj+1] += delta[:,:,i,j] * (1-li)*(1-lj)
        return [delta_in[:,:,:-1,:-1]]

class Conv():
    #ref : http://blog.csdn.net/l_b_yuan/article/details/64927643
    def __init__(self, name, inp_channel, out_channel, kernel_shape, stride, padding = (0,0), W = None, b = None):
        self.name = name
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.padding = padding
        self.inp_channel = inp_channel
        self.out_channel = out_channel
        if W is None:
            kernel_size = inp_channel * kernel_shape[0] * kernel_shape[1]
            kernel = np.random.uniform(-1,1,(out_channel, inp_channel, kernel_shape[0], kernel_shape[1]))/np.sqrt(kernel_size)
            self.W = kernel
        else:
            self.W = W
        if b is None:
            self.b = np.zeros((out_channel,))
        else:
            self.b = b

    def __call__(self, inputs):
        img = inputs[0]
        N,C,H,W = img.shape
        F,_,HH,WW = self.W.shape
        SH, SW = self.stride
        PH, PW = self.padding
        OH = 1 + (H + 2 * PH - HH) // SH
        OW = 1 + (W + 2 * PW - WW) // SW
        x_pad = np.zeros((N, C, H+2*PH, W+2*PW))
        x_pad[:,:,PH:PH+H, PW:PW+W] = img
        out = np.zeros((N, F, OH, OW))
        for f in range(F):
            for i in range(OH):
                for j in range(OW):
                    out[:,f,i,j] = np.sum(x_pad[:,:,i*SH:i*SH+HH,j*SW:j*SW+WW]*self.W[f,:,:,:], axis = (1,2,3))
            out[:,f,:,:] += self.b[f]
        return out

    def backprop(self, inputs, delta, lr,weight_decay):
        img = inputs[0]
        N,F,H1,W1 = delta.shape
        N,C,H,W = img.shape
        HH, WW = self.kernel_shape
        SH, SW = self.stride
        PH, PW = self.padding
        dx = np.zeros_like(img)
        dw = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        x_pad = np.pad(img, [(0,0),(0,0),(PH,PH),(PW,PW)], 'constant')
        dx_pad = np.pad(dx, [(0,0),(0,0),(PH,PH),(PW,PW)], 'constant')
        db = np.sum(delta, axis = (0,2,3))
#        for n in range(N):
        for i in range(H1):
            for j in range(W1):
                x_window = x_pad[:,:,i*SH:i*SH+HH, j*SW:j*SW+WW]
                for f in range(F):
                    dw[f:f+1] += (x_window * delta[:,f:f+1,i:i+1,j:j+1]).sum(axis=0)
                    dx_pad[:,:,i*SH:i*SH+HH,j*SW:j*SW+WW] += self.W[f:f+1] * delta[:,f:f+1,i:i+1,j:j+1]
        delta_in = dx_pad[:,:,PH:PH+H,PW:PW+W]
        dw /= N
        dw += weight_decay * self.W
        db /= N
        db += weight_decay * self.b
        self.W -= lr * dw
        self.b -= lr * db
        return [delta_in]


class Pooling():
    def __init__(self, name, kernel_shape, stride, mode):
        self.name = name
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.mode = mode

    def __call__(self, inputs):
        imgs = inputs[0]
        data = []
        for i in range(0, imgs.shape[2], self.stride[0]):
            if i + self.kernel_shape[0] > imgs.shape[2]:
                break
            for j in range(0, imgs.shape[3], self.stride[1]):
                if j + self.kernel_shape[1] > imgs.shape[3]:
                    break
                data.append(imgs[:,:,i:i+self.kernel_shape[0], j:j+self.kernel_shape[1]].reshape((imgs.shape[0], imgs.shape[1], -1)))
            data = np.array(data).transpose((1,2,0,3))
            if self.mode == 'max':
                re = np.max(data, axis=3)
            elif self.mode == 'avg':
                re = np.mean(data, axis = 3)
        re = re.reshape((imgs.shape[0], imgs.shape[1], (imgs.shape[2]-self.kernel_shape[0])//self.stride[0]+1, (imgs.shape[3]-self.kernel_shape[1])//self.stride[1]+1))
        return re

    def backprop(self, inputs, delta, lr,weight_decay):
        #Not Implemented
        pass

class CrossEntropy(MultiOprs):
    def __call__(self, inputs):
        pred, label = inputs
        pred = np.exp(pred - pred.max(axis=1, keepdims=True))
        pred = pred / pred.sum(axis = 1, keepdims = True)
        return (-np.log(pred) * label).mean()

    def backprop(self, inputs, delta, lr, weight_decay):
        pred, label = inputs
        pred = pred - label
        return [pred, np.zeros_like(label)]
        
func_dict = {'SUM':Sum, 'ADD': Add, 'SUB': Sub, 'MUL':Mul, 'DIV': Div, 'CONV': Conv, 'FC': FC, 'POOL': Pooling, 'CONCAT': Concat, 'EXP': Exp, 'POW': Pow, 'RELU': ReLU, 'SIGMOID': Sigmoid, 'TANH': Tanh, 'RESHAPE':Reshape,'US':UpSampling, 'BCE': CrossEntropy, }


