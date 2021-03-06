import cv2
from image_process import edge_det, DCT, rowDelta, LBP 
import numpy as np 
import os

def asInt8(img):
    img[img < 0] = 0
    img[img > 255] = 255
    return img.astype('uint8')

def init_base(shape):
    tmp = np.zeros(shape, dtype=int)
    for i in range(11):
        img = cv2.imread('../data/frame%s.jpg'%i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tmp += img
    tmp = tmp / 11.0
    tmp = asInt8(tmp)
    return tmp

class VideoPlay:
    def __init__(self, src, step=32):
        self.src = cv2.VideoCapture(src)
        self.window = cv2.namedWindow('x')
        self.step = step
        self.dct = DCT(self.step, self.step)
        self.dct_all = None
        #self.base = init_base(self.shape)

    @property
    def shape(self):
        if self.src.isOpened():
            ret, frame = self.src.read()
            return frame.shape[:2]
        else:
            return None

    def play_LBP(self):
        H, W = self.shape
        print(H, W)
        dct = DCT(h=H,w=W)
        while self.src.isOpened():
            ret, frame = self.src.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                img = LBP(frame)
                img = dct.H_filter(img, 150) 
                img = np.concatenate([img, frame], axis=1) 
                cv2.imshow('x', img)
                key = cv2.waitKey(30)
                if key == ord('q'):
                    return 1
        return -1

    def play_all(self):
        H, W = self.shape
        print(H, W)
        if self.dct_all is None:
            self.dct_all = DCT(H, W)
        tmp = []
        while self.src.isOpened():
            ret, frame = self.src.read()
            alpha = 1
            if ret:
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                tmp.append(frame)
                if len(tmp) > 5:
                    tmp.pop(0)
                img = np.array(tmp).mean(axis=0).astype('uint8')
                cv2.imshow('x', img)
                key = cv2.waitKey(30)
                if key == ord('q'):
                    return 1
        return -1


    def play(self):
        H, W = self.shape
        print(H, W)
        hc = H // self.step
        wc = W // self.step
        count = 0
        frames = []
        while self.src.isOpened():
            ret, frame = self.src.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)
                if len(frames) != 5:
                    continue
                frame = np.mean(np.array(frames,dtype='float32'), axis=0).astype('uint8')
                thr = 50
                count += 1
                if count // 50 % 2 == 0:
                    delta = np.random.normal(scale=10, loc=0, size=(H, W))
                else:
                    delta = np.random.randint(-1, 2, (H, W)) * 10
                frame0 = frame + delta 
                frame0 = asInt8(frame0)
                outframe = np.zeros((H,W), dtype='uint8')
                for i in range(hc):
                    for j in range(wc):
                        tmpframe = frame0[i*self.step:(i+1)*self.step, j*self.step:(j+1)*self.step]
                        frame1 = self.dct.inference(tmpframe)
                        frame1[np.abs(frame1) < thr] = 0
                        frame2 = self.dct.invert(frame1)
                        frame2 = asInt8(frame2)
                        outframe[i*self.step:(i+1)*self.step, j*self.step:(j+1)*self.step] = frame2

                outframe2 = np.zeros((H,W), dtype='uint8')
                for i in range(hc):
                    for j in range(wc):
                        tmpframe = frames[0][i*self.step:(i+1)*self.step, j*self.step:(j+1)*self.step]
                        frame3 = self.dct.inference(tmpframe)
                        frame3[np.abs(frame3) < thr] = 0
                        frame4 = self.dct.invert(frame3)
                        frame4 = asInt8(frame4)
                        outframe2[i*self.step:(i+1)*self.step, j*self.step:(j+1)*self.step] = frame4

                frames = []

                frame = np.concatenate([frame, outframe, outframe2], axis=1)
                cv2.imshow('x', frame)
                key = cv2.waitKey(30)
                if key == ord('q'):
                    return 1
        return -1

if __name__ == '__main__':
    play = VideoPlay(0, 160)
    L = play.play_LBP()
