import numpy as np 
import cv2
import sys
import os
from tqdm import tqdm
from functools import reduce
from decorators import timer

def asInt8(img):
    img = img.clip(0,255)
    return img.astype('uint8')

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

def blur(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img, dtype='int')
    re = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            tmp = np.abs(img[max(i-5,0):i+6, max(j-5,0):j+6])
            re[i][j] = tmp.mean()
    return re

def gamma(img, g):
    return (((img.astype('float32')/256.0)**g)*256.0).astype('uint8')

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

if __name__ == '__main__':
    a = DCT(480, 640)
    b = a.inference(np.ones((480, 640)))
    c = a.invert(b)
    from IPython import embed
    embed()
