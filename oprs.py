import numpy as np
import cv2
class MultiOprs():
    def __init__(self, name):
        self.name = name

class Sum(MultiOprs):
    def __call__(self, inputs):
        A = inputs[0]
        return A.sum()

    def backprop(self, inputs, delta, lr):
        A = inputs[0]
        return [np.ones(A.shape)]

class Add(MultiOprs):
    def __call__(self, inputs):
        A, B = inputs
        return A + B

    def backprop(self, inputs, delta, lr):
        return [delta, delta]

class Sub(MultiOprs):
    def __call__(self, inputs):
        A, B = inputs
        return A - B

    def backprop(self, inputs, delta, lr):
        return [delta, -delta]

class Mul(MultiOprs):
    def __call__(self, inputs):
        A, B = inputs
        return A * B

    def backprop(self, inputs, delta, lr):
        A, B = inputs
        return [delta * B, delta * A]

class Div(MultiOprs):
    def __call__(self, inputs):
        A, B = inputs
        return A / B

    def backprop(self, inputs, delta, lr):
        A, B = inputs
        return [delta / B, - delta * A / (B**2)]

class Exp(MultiOprs):
    def __call__(self, inputs):
        _ = inputs[0]
        return np.exp(_)

    def backprop(self, inputs, delta, lr):
        _ = inputs[0]
        return [delta * np.exp(_)]

class Pow():
    def __init__(self, name, k):
        self.name = name
        self.k = k
        pass

    def __call__(self, inputs):
        _ = inputs[0]
        return _ ** self.k

    def backprop(self, inputs, delta, lr):
        _ = inputs[0]
        return [self.k * delta * (_ ** (self.k - 1))]

class Concat():
    def __init__(self, name, axis):
        self.axis = axis
        self.name = name
        pass

    def __call__(self, inputs):
        return np.concatenate(inputs, axis = self.axis)

    def backprop(self, inputs, delta, lr):
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

    def backprop(self, inputs, delta, lr):
        _ = inputs[0]
        re = delta * ((_ > 0).astype('int'))
        return [re]

class Sigmoid(MultiOprs):
    def __call__(self, inputs):
        e = np.exp(inputs[0])
        return 1 - 1.0/(1+e)

    def backprop(self, inputs, delta, lr):
        e = np.exp(inputs[0])
        sig = 1 - 1.0/(1+e)
        return [sig*(1-sig)*delta]

class Tanh(MultiOprs):
    def __call__(self, inputs):
        return np.tanh(inputs[0])

    def backprop(self, inputs, delta, lr):
        return [(1-np.tanh(inputs[0])**2)*delta]

class Conv():
    #ref : http://blog.csdn.net/l_b_yuan/article/details/64927643
    def __init__(self, name, inp_channel, out_channel, kernel_shape, stride, W = None, b = None):
        self.name = name
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.inp_channel = inp_channel
        self.out_channel = out_channel
        if W is None:
            kernel_size = inp_channel * kernel_shape[0] * kernel_shape[1]
            kernel = np.random.uniform(-1,1,(kernel_size, out_channel))/np.sqrt(kernel_size)
            self.W = kernel
        else:
            self.W = W
        if b is None:
            self.b = np.zeros((1, out_channel))
        else:
            self.b = b

    def transfor(self, img):
        data = []
        for i in range(0, img.shape[2], self.stride[0]):
            if i + self.kernel_shape[0] > img.shape[2]:
                continue
            for j in range(0, img.shape[3], self.stride[1]):
                if j + self.kernel_shape[1] > img.shape[3]:
                    break
                data.append(img[:,:,i:i+self.kernel_shape[0],j:j+self.kernel_shape[1]].reshape(img.shape[0], -1))
        data = np.array(data).transpose((1,0,2))
        return data

    def __call_(self, inputs):
        img = inputs[0]
        data = self.transfor(img)
        re = data @ self.W + self.b
        re = (re.transpose(0,2,1)).reshape((img.shape[0], self.out_channel, (img.shape[2] - self.kernel_shape[0])//self.stride[0]+1, (img.shape[3]-self.kernel_shape[1])//self.stride[1]+1))
        return re

    def backprop(self, inputs, delta, lr):
        # Not Implemented
        pass

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

    def backprop(self, inputs, delta, lr):
        grad_weights = inputs[0].T @ delta / inputs[0].shape[0]
        grad_bias = delta.mean(axis = 0)
        inp_delta = delta @ self.W.T
        self.W -= lr * grad_weights
        self.b -= lr * grad_bias
        return [inp_delta]

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
        re = re.reshape((imgs.shape[0], imgs.shape[1], (imgs.shape[2]-self.kernel_shape[0])//stride[0]+1, (imgs.shape[3]-kernel_shape[1])//stride[1]+1))
        return re

    def backprop(self, inputs, delta, lr):
        #Not Implemented
        pass



func_dict = {'SUM':Sum, 'ADD': Add, 'SUB': Sub, 'MUL':Mul, 'DIV': Div, 'CONV': Conv, 'FC': FC, 'POOL': Pooling, 'CONCAT': Concat, 'EXP': Exp, 'POW': Pow, 'RELU': ReLU, 'SIGMOID': Sigmoid, 'TANH': Tanh, }


