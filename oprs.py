import numpy as np
import cv2

class conv():
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
                data.append(img[:,:,i:i+self.kernel_shape[0],j:j+self.kernel_shape[1]].reshape((img.shape[0],-1)))
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
        else:
            kernel=self.kernel
            bias = self.bias
        re = data @ kernel + bias
        re = (re.transpose(0,2,1)).reshape((img.shape[0], self.out_channel, (img.shape[2] - self.kernel_shape[0])//self.stride[0]+1, (img.shape[3]-self.kernel_shape[1])//self.stride[1]+1))
        return re

    def backprop(self, img, delta, lr):
        data = self.transfor(img)
        tmp = delta.reshape(img.shape[0], -1, self.out_channel).transpose(0,2,1)
        grad_kernel = data.transpose(0,2,1) @ delta
        grad_bias = delta.mean(axis = 0)
        inp_delta = (delta @ self.kernel.T).reshape(img.shape)
        self.kernel -= lr * grad_kernel
        self.bias -= lr * grad_bias
        return inp_delta


class fc():
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
