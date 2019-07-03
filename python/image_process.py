import numpy as np 
import cv2
import sys
import os
from tqdm import tqdm
from functools import reduce
import itertools
from decorators import timer
kernel = np.random.uniform(0.09, 0.11, (10, 10))
kernel = kernel / kernel.sum()
H = 640
W = 480

idx = np.array(list(itertools.product(range(H), range(W)))).reshape(H, W, 2)
@timer
def frosted_glass_gen(src, scale=5):
    kernel = np.random.uniform(0.09, 0.11, (scale, scale))
    kernel = kernel / kernel.sum()
    rows, cols, _ = src.shape
    assert rows == H and cols == W
    offsets = np.random.randint(3, 6)
    random_num = 0
    off = np.random.randint(0, offsets, (rows, cols, 1))
    target = idx + off
    dst = np.zeros((H, W, 3), dtype='uint8')
    for y in range(rows-offsets):
        for x in range(cols-offsets):
            dst[y, x] = src[target[y,x,0], target[y,x,1]]
    dst = cv2.filter2D(dst, -1, kernel)
    dst = cv2.filter2D(dst, -1, kernel)
    return dst

@timer
def fgg_v2(src, scale=5):
    rows, cols, _ = src.shape
    assert rows == H and cols == W
    offsets = np.random.randint(3, 6)
    random_num = 0
    off = np.random.randint(0, offsets, (rows, cols, 2))
    target = (idx + off).astype('float32')
    dst = np.zeros((H, W, 3), dtype='uint8')
    h = H-offsets
    w = W-offsets
    base = np.random.randint(100, 130, (H,W,3), dtype='uint8')
    dst[:h, :w] = cv2.remap(src, target[:h,:w,1], target[:h,:w,0], cv2.INTER_LINEAR)
    alpha = np.random.uniform(0.5, 0.7)
    dst = cv2.filter2D(dst, -1, kernel)
    dst = cv2.filter2D(dst, -1, kernel)
    dst = asInt8(alpha * base + (1-alpha)*dst)
    return dst

@timer
def fgg_v3(src, scale=5):
    rows, cols, _ = src.shape
    assert rows == H and cols == W
    offsets = np.random.randint(3, 6)
    random_num = 0
    off = np.random.randint(0, offsets, (rows, cols, 1))
    target = (idx + off).astype('float32')
    dst = np.zeros((H, W, 3), dtype='uint8')
    h = H-offsets
    w = W-offsets
    dst[:h, :w] = cv2.remap(src, target[:h,:w,1], target[:h,:w,0], cv2.INTER_LINEAR)
    base = np.random.randint(100, 130, (H,W,3), dtype='uint8')
    alpha = np.random.uniform(0.5, 0.6)
    dst = cv2.filter2D(dst, -1, kernel)
    dst = cv2.filter2D(dst, -1, kernel)
    dst = asInt8(alpha * base + (1-alpha)*dst)
    return dst

def asInt8(img):
    img = img.clip(0,255)
    return img.astype('uint8')

def rowDelta(img):
    img = img.astype(int)
    a = (img**2).sum(axis=1, keepdims=True)
    b = img @ img.T
    c = (a + a.T - 2*b) ** 0.5
    d = asInt8(c / c.max() * 255)
    return d

class DCT:
    @timer
    def __init__(self, h, w):
        self.h = h
        self.w = w
        self.alpha_p = np.ones((h,)) * np.sqrt(2/h)
        self.alpha_p[0] = np.sqrt(1/h)
        self.alpha_q = np.ones((w,)) * np.sqrt(2/w)
        self.alpha_q[0] = np.sqrt(1/w)
        arr1 = np.array(range(h)).reshape(h,1)
        arr2 = np.array(range(w)).reshape(w,1)
        self.cos_p = np.cos(np.pi * (arr1 @ arr1.T * 2 + arr1) / (2 * h))
        self.cos_q = np.cos(np.pi * (arr2 @ arr2.T * 2 + arr2) / (2 * w))
        
    def inference(self, img):
        assert img.shape[:2] == (self.h, self.w)
        out = self.cos_p @ (img @ self.cos_q.T)
        out = out * (self.alpha_p.reshape((self.h,1)) @ self.alpha_q.reshape((1,self.w)))
        return out

    def invert(self, img):
        assert img.shape[:2] == (self.h, self.w)
        out = (self.cos_p * self.alpha_p.reshape(self.h,1)).T @ (img @ (self.cos_q*self.alpha_q.reshape(self.w,1)))
        return out

    def L_filter(self, img, thr):
        assert img.shape[:2] == (self.h, self.w)
        freq = self.inference(img)
        freq[np.abs(freq) < thr] = 0
        img = self.invert(freq)
        img = asInt8(img)
        return img

    def H_filter(self, img, thr):
        assert img.shape[:2] == (self.h, self.w)
        freq = self.inference(img)
        freq[freq > thr] = thr
        freq[freq < -thr] = -thr
        img = self.invert(freq)
        img = asInt8(img)
        return img

def YUV2PNG(h, w, msg):
    img_y=np.fromstring(msg[:h*w],dtype='uint8').reshape((h,w)).astype('int32')
    img_v=np.fromstring(msg[h*w:h*w+h*w//2:2],dtype='uint8').reshape((h//2,w//2)).astype('int32')
    img_u=np.fromstring(msg[h*w+1:h*w+h*w//2:2],dtype='uint8').reshape((h//2,w//2)).astype('int32')
    ruv=((359*(img_v-128))>>8)
    guv=-1*((88*(img_u-128)+183*(img_v-128))>>8)
    buv=((454*(img_u-128))>>8)
    ruv=np.repeat(np.repeat(ruv,2,axis=0),2,axis=1)
    guv=np.repeat(np.repeat(guv,2,axis=0),2,axis=1)
    buv=np.repeat(np.repeat(buv,2,axis=0),2,axis=1)
    img_r=(img_y+ruv).clip(0,255).astype('uint8')
    img_g=(img_y+guv).clip(0,255).astype('uint8')
    img_b=(img_y+buv).clip(0,255).astype('uint8')
    img=np.dstack([img_b[:,:,None],img_g[:,:,None],img_r[:,:,None]])
    img=img.transpose((1,0,2))[::-1].copy()
    return img

@timer
def LBP(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = img.shape
    re = np.zeros(img.shape)
    z1 = np.zeros((H, 1))
    z2 = np.zeros((1, W))
    
    a1 = (np.concatenate([z2, np.concatenate([z1[1:], img[:H-1,:W-1]], axis=1)], axis=0) >= img)*1
    a2 = (np.concatenate([z2, img[:H-1]], axis=0) >= img)*2
    a3 = (np.concatenate([z2, np.concatenate([img[:H-1, 1:], z1[1:]], axis=1)], axis=0) >= img)*4
    a4 = (np.concatenate([img[:,1:], z1], axis=1) >= img)*8
    a5 = (np.concatenate([np.concatenate([img[1:,1:], z1[1:]], axis=1), z2], axis=0) >= img)*16
    a6 = (np.concatenate([img[1:], z2], axis=0) >= img)*32
    a7 = (np.concatenate([np.concatenate([img[1:, :W-1], z1[1:]], axis=1), z2], axis=0) >= img)*64
    a8 = (np.concatenate([z1, img[:,:W-1]], axis=1) >= img)*128

    re = a1+a2+a3+a4+a5+a6+a7+a8
    return re.astype('uint8')


@timer
def edge_det(img, norm=False):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img, dtype='float32')
    if norm:
        mean = img.mean()
        std = img.std()
        img = (img - mean) / std
    H, W = img.shape
    re = np.zeros(img.shape)
    z1 = np.zeros((H, 1))
    z2 = np.zeros((1, W))
    
    a1 = np.abs(np.concatenate([z2, np.concatenate([z1[1:], img[:H-1,:W-1]], axis=1)], axis=0) - img)
    a2 = np.abs(np.concatenate([z2, img[:H-1]], axis=0) - img)
    a3 = np.abs(np.concatenate([z2, np.concatenate([img[:H-1, 1:], z1[1:]], axis=1)], axis=0) - img)
    a4 = np.abs(np.concatenate([img[:,1:], z1], axis=1) - img)
    a5 = np.abs(np.concatenate([np.concatenate([img[1:,1:], z1[1:]], axis=1), z2], axis=0) - img)
    a6 = np.abs(np.concatenate([img[1:], z2], axis=0) - img)
    a7 = np.abs(np.concatenate([np.concatenate([img[1:, :W-1], z1[1:]], axis=1), z2], axis=0) - img)
    a8 = np.abs(np.concatenate([z1, img[:,:W-1]], axis=1) - img)

    re = reduce(np.maximum, [0, a1, a2, a3, a4, a5, a6, a7, a8])
    re = (re / re.max()) * 255
    return re.astype('uint8')

@timer
def blur(img, kernel_size=5):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
    dst_img = cv2.filter2D(img, -1, kernel)
    return dst_img

@timer
def gamma(img, g):
    return adjust_gamma(img, gamma=g)

@timer
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

@timer
def lighter(img):
    h, w = img.shape[:2]
    mean = img.mean()
    h = np.log(mean) / np.log(128)
    return adjust_gamma(img, h)

@timer
def add_lines(img, width=10):
    pattern = np.array(range(img.shape[1]))
    pattern = np.cos(pattern / (width*4)*2*np.pi) * 20
    pattern = pattern.reshape((1, img.shape[1], ) + (1,)*(img.ndim-2))
    img = np.clip(img + pattern, 0, 255).astype('uint8')
    return img

def convert(h, w, msg):
    img_y=np.fromstring(msg[:h*w],dtype='uint8').reshape((h,w))
    img_cr=cv2.resize(np.fromstring(msg[h*w:h*w+h*w//2:2],dtype='uint8').reshape((h//2,w//2)),(w,h),interpolation=0)
    img_cb=cv2.resize(np.fromstring(msg[h*w+1:h*w+h*w//2:2],dtype='uint8').reshape((h//2,w//2)),(w,h),interpolation=0)
    try:
        convert=cv2.cv.CV_YCrCb2BGR
    except:
        convert=cv2.COLOR_YCrCb2BGR
    img=cv2.cvtColor(cv2.merge([img_y,img_cr,img_cb]),convert)
    img=img.transpose((1,0,2))[::-1,::-1].copy()
    return img

def convert_dir(argv):
    if len(argv) < 2:
        sys.stderr.write("{} <image_dir>".format(argv[0]))
        sys.exit(1)
    input_image_dir = argv[1]

    image_files = os.listdir(input_image_dir)
    for image_file in image_files:
        if image_file[-3:] in ['jpg','png','bmp']:
            continue
        try:
            image_path = os.path.join(input_image_dir, image_file)
            splits = image_file.split(".")
            h = 480#int(splits[-1])
            w = 640#int(splits[-2])
            with open(image_path, "rb") as fn:
                content = fn.read()
                out_img = convert_fhq(h, w, content)
                cv2.imwrite(image_path + ".png", out_img)
        except Exception as e:
            sys.stderr.write("Failed to handle {}\n".format(image_file))
            sys.stderr.write("Exception: {}\n".format(e))

class StripeMask:
    def __init__(self, range_angle=None, range_size=None, range_rate=None):
        self.range_angle = self.float2range_list(range_angle or [-15, 15])
        self.range_size = self.float2range_list(range_size or [20, 20+1e-9])
        self.range_rate = self.float2range_list(range_rate or [0.3, 0.3+1e-9])
        self.shift = [0, 2*np.pi]

    def __call__(self, img):
        n_stripe = self.gen_random(self.range_size)
        angle = self.gen_random(self.range_angle)/180*np.pi
        rate = self.gen_random(self.range_rate)
        shift = self.gen_random(self.shift)
        deeps = self.gen_mask_angle(img, n_stripe, angle, shift, rate)
        img = np.clip(img * deeps, 0, 255).astype(img.dtype)

        return img

    def gen_mask_angle(self, img, n_stripe, angle=0, shift=0, rate=0.3):
        assert 2 <= len(img.shape) <= 3
        len_shape = len(img.shape)
        x = np.linspace(0, 2*np.pi*n_stripe, img.shape[1])
        y = np.linspace(0, 2*np.pi*n_stripe, img.shape[0])
        X, Y = np.meshgrid(x, y)
        shape = X.shape

        deeps = (np.sin((np.cos(angle)*X+np.sin(angle)*Y+shift)) + 1)/2*rate+(1-rate)
        if len_shape == 3:
            deeps = deeps.reshape(*img.shape[:2], 1)
        return deeps

    @staticmethod
    def float2range_list(x):
        x = [x, x+1e-9] if isinstance(x, int) or isinstance(x, float) else x
        assert x[1] > x[0]
        return x

    def gen_random(self, range_x):
        return np.random.uniform(range_x[0], range_x[1])

stripe_range_angle, stripe_range_size, stripe_range_rate = [-15, 15], [10, 30], [0.05, 0.2]
stripe_mask_aug = StripeMask(stripe_range_angle, stripe_range_size, stripe_range_rate)

@timer
def add_lines_v2(img):
    img_new = stripe_mask_aug(img)
    return img_new

if __name__ == '__main__':
    count = 0
    def cvt_img(img, name):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('%s.png'%name, img)
        img = add_lines(img, width = 10 - count)
        cv2.imwrite('%s_2.png'%name, img)
        img = blur(img, kernel_size=10)
        cv2.imwrite('%s_3.png'%name, img)
        cv2.imshow('x', img)
        cv2.waitKey(0)

    img = cv2.imread('/home/liuyu/Pictures/ckq_wyz/compareFailed/09-09-20_29_43.187.nv21.640.480.png')
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    tmp = add_lines_v2(img)
    cv2.imshow('x', tmp)
    cv2.waitKey(0)

    tmp = add_lines(img)
    cv2.imshow('y', tmp)
    cv2.waitKey(0)

    exit(0)
    cvt_img(img, 'test')
    count +=1 
    img = cv2.imread('/home/liuyu/Pictures/ckq_wyz/compareFailed/09-09-20_29_58.806.nv21.640.480.png')
    cvt_img(img, 'test2')
