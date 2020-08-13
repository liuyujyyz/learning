import cv2
import numpy as np
import scipy.signal as sign
from .decorators import timer


def as_img(ndarray):
    img = np.clip(ndarray, 0, 255)
    return img.astype('uint8')


def normalize(img):
    """
    img: hwc
    """
    h, w, c = img.shape
    img = img.reshape(h*w,c)
    img = (img - img.mean(axis=0, keepdims=True)) / img.std(axis=0, keepdims=True)
    return img.reshape(h, w, c)


@timer
def conv2d(img, kernel):
    """
    kernel: iohw
    img: hwc
    """
    in_, out_, _, _ = kernel.shape
    re_ = []
    for j in range(out_):
        tmp = 0
        for i in range(in_):
            tmp += sign.convolve2d(img[:,:,i], kernel[i,j], mode='same')
        re_.append(tmp)
    return np.array(re_).transpose(1,2,0)


def slice_window(img, kernel, stride):
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if isinstance(stride, int):
        stride = (stride, stride)
    if img.ndim == 3:
        h, w, _ = img.shape
    else:
        h, w = img.shape
    
    out_ = []
    for i in range(0, h-kernel[0], stride[0]):
        for j in range(0, w-kernel[1], stride[1]):
            out_.append(img[i:i+kernel[0], j:j+kernel[1]])

    out_ = np.array(out_)
    return out_


def formula_size(img, shorter=256):
    if img.ndim == 3:
        h, w, c = img.shape
    else:
        h, w = img.shape

    if h > w:
        h = h * shorter // w
        w = shorter
    else:
        w = w * shorter // h
        h = shorter

    return cv2.resize(img, (w, h))


def gamma_lighter(img, avg = 128):
    if avg > 1:
        avg = avg / 255
    img = img.astype('float32') / 255
    s = img.mean()
    gamma = np.log(avg) / np.log(s)
    print(gamma)
    return as_img(img**gamma * 255)

def norm(arr):
    return ((arr - arr.mean()) / arr.std() + 2)*0.1 + arr / 255 * 0.6 

def norm_uv(arr):
    return ((arr - arr.mean()) / arr.std() + 3) / 6 * 0.2 + arr / 255 * 0.8

def lighter(img, channel_wise=True):
    img = img.astype('float32')
    if channel_wise:
        A = norm(img[:,:,0])
        B = norm(img[:,:,1])
        C = norm(img[:,:,2])
        img[:,:,0] = A
        img[:,:,1] = B
        img[:,:,2] = C
    else:
        img = norm(img)
    img = img * 255
    return as_img(img)


def lighter_yuv(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = img.astype('float32')
    Y = norm(img[:,:,0])
    U = norm_uv(img[:,:,1])
    V = norm_uv(img[:,:,2])
    img[:,:,0] = Y
    img[:,:,1] = U
    img[:,:,2] = V
    img = as_img(img * 255)
    img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    return img


if __name__ == '__main__':
    pass
