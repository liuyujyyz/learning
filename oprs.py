import numpy as np
import cv2

class conv():
    def __init__(out_channel, kernel_shape, stride, kernel_weights = None):
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.out_channel = out_channel
        self.weights = kernel_weights

    def forward(self, img):
        data = []
        for i in range(0, img.shape[2], stride[0]):
            if i + self.kernel_shape[0] > img.shape[2]:
                continue
            for j in range(0, img.shape[3], stride[1]):
                if j + self.kernel_shape[1] > img.shape[3]:
                    break
                data.append(img[:,:,i:i+self.kernel_shape[0],j:j+self.kernel_shape[1]].reshape((img.shape[0],-1)))
        data = np.array(data).transpose((1,0,2))
        if self.weights is None:
            kernel = []
            kernel_size = img.shape[1] * self.kernel_shape[0] * self.kernel_shape[1]
            for i in range(self.out_channel):
                kernel.append(np.random.normal(0,1,(kernel_size,)))
            kernel = np.array([np.array(kernel).T])
            self.weights = kernel
        else:
            kernel=self.weights
        re = data @ kernel
        re = (re.transpose(0,1,2)).reshape((img.shape[0], self.out_channel, (img.shape[2] - self.kernel_shape[0])//self.stride[0]+1, (img.shape[3]-self.kernel_shape[1])//self.stride[1]+1))
        return re

    def backprop(self, img, grad):

