import numpy as np
import cv2

class Sigmoid():
    def __init__(self):
        pass

    def forward(self, inputs):
        e = np.exp(inputs)
        return 1 - 1.0/(1+e)

    def backprop(self, inputs, delta):
        e = np.exp(inputs)
        sig = 1 - 1.0/(1+e)
        return sig*(1-sig)*delta

class Tanh():
    def __init__(self):
        pass

    def forward(self, inputs):
        return np.tanh(inputs)

    def backprop(self, inputs, delta):
        return (1-np.tanh(inputs)**2)*delta

class Conv():
    #ref : http://blog.csdn.net/l_b_yuan/article/details/64927643
    def __init__(self, inp_channel, out_channel, kernel_shape, stride, kernel_weights = None, bias_weights = None):
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.inp_channel = inp_channel
        self.out_channel = out_channel
        self.kernel = kernel_weights
        self.bias = bias_weights

    def transfor(self, img):
        data = []
        for i in range(0, img.shape[2], self.stride[0]):
            if i + self.kernel_shape[0] > img.shape[2]:
                continue
            for j in range(0, img.shape[3], self.stride[1]):
                if j + self.kernel_shape[1] > img.shape[3]:
                    break
                data.append(img[:,:,i:i+self.kernel_shape[0],j:j+self.kernel_shape[1]])
        data = np.array(data).transpose((1,0,2))
        return data

    def forward(self, img):
        data = self.transfor(img)
        if self.kernel is None:
            kernel = []
            kernel_size = self.inp_channel * self.kernel_shape[0] * self.kernel_shape[1]
            kernel = np.random.uniform(-1,1,(kernel_size, self.out_channel))/np.sqrt(kernel_size)
            self.kernel = kernel
            bias = np.zeros((1, self.out_channel))
            self.bias = bias
        else:
            kernel=self.kernel
            bias = self.bias
        re = data @ kernel + bias
        re = (re.transpose(0,2,1)).reshape((img.shape[0], self.out_channel, (img.shape[2] - self.kernel_shape[0])//self.stride[0]+1, (img.shape[3]-self.kernel_shape[1])//self.stride[1]+1))
        return re

    def backprop(self, img, delta, lr):
        data = self.transfor(img)
        grad_bias = delta.mean(axis = (0,2,3))
        """
        grad_kernel = (data.transpose(0,2,1) @ tmp).mean(axis=0)
        inp_delta = (tmp @ self.kernel.T).transpose((1,0,2))
        inp_delta = inp_delta.reshape(inp_delta.shape[0], inp_delta.shape[1], self.inp_channel, self.kernel_shape[0], self.kernel_shape[1])
        out = np.zeros(img.shape)
        count = 0
        """
        delta_inp = np.zeros_like(img)
        grad_kernel = np.zeros_like(self.kernel)
        for i in range(delta.shape[2]):
            for j in range(delta.shape[3]):
                x_window = img[:,:,i*self.stride[0]:i*self.stride[0]+self.kernel_shape[0], j*self.stride[1]:j*self.stride[1]+self.kernel_shape[1]]
                for f in range(self.out_channel):
                    grad_kernel[:,f] += (x_window * delta[:,f:f+1,i:i+1,j:j+1]).sum(axis=(0,2,3)).reshape(-1)
                    delta_inp[:,:,i*self.stride[0]:i*self.stride[0]+self.kernel_shape[0],j*self.stride[1]:j*self.stride[1]+self.kernel_shape[1]] += self.kernel[:,f].reshape(1,1,self.kernel_shape[0], self.kernel_shape[1]) * delta[:,f:f+1,i:i+1,j:j+1]
        
        grad_kernel /= img.shape[0]
        self.kernel -= lr * grad_kernel
        self.bias -= lr * grad_bias
        return delta_inp.mean(axis=0)


class FC():
    def __init__(self, inp_size, out_size, weights = None, bias = None):
        self.inp_size = inp_size
        self.out_size = out_size
        if weights is None:
            self.weights = np.random.uniform(-1,1,(inp_size, out_size))/np.sqrt(inp_size)
        else:
            self.weights = weights
        if bias is None:
            self.bias = np.zeros((1, out_size))
        else:
            self.bias = bias
        assert self.weights.shape[0] == inp_size and self.weights.shape[1] == out_size; 'not matching'

    def forward(self, inputs):
        return inputs@self.weights + self.bias

    def backprop(self, inputs, delta, lr):
        grad_weights = inputs.T @ delta / inputs.shape[0]
        grad_bias = delta.mean(axis = 0)
        inp_delta =  delta @ self.weights.T
        self.weights -= lr * grad_weights
        self.bias -= lr * grad_bias
        return inp_delta

class Pooling():
    def __init__(self, kernel_shape, stride, mode):
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.mode = mode

    def forward(self, imgs):
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
