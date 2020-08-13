import cv2
import numpy as np 
import os
from utils.libs.imgproc import conv2d, normalize


np.random.seed(0)


def asInt8(img):
    img = np.clip(img, 0, 255)
    return img.astype('uint8')


class VideoPlay:
    def __init__(self, src, kernel=None, depth=1, size=3):
        self.src = cv2.VideoCapture(src)
        self.window = cv2.namedWindow('x')
        if kernel is not None:
            assert kernel.ndim == 5
            depth = kernel.shape[0]
            size = kernel.shape[-1]
        self.filter = kernel
        self.depth = depth
        self.size = size

    @property
    def shape(self):
        if self.src.isOpened():
            ret, frame = self.src.read()
            return frame.shape[:2]
        else:
            return None

    def play_otsu(self):
        while self.src.isOpened():
            ret, frame = self.src.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5,5), 0)
                ret, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
                show = np.concatenate([gray, mask], axis=1)
                cv2.imshow('x', show)
                key = cv2.waitKey(30)
                if key == ord('q'):
                    return 1
        return -1

    def play(self):
        if self.filter is None:
            kernel = np.random.normal(0, 1, (self.depth, 3,3,self.size, self.size)) / np.sqrt(27)
        else:
            kernel = self.filter
        while self.src.isOpened():
            ret, frame = self.src.read()
            if ret:
                frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
                modify = (frame - 128.0) / 128.0
                alpha = 0.05
                for i in range(self.depth):
                    modify = conv2d(modify, kernel[i]) 
                    modify = normalize(modify)
                    modify[modify<0] *= (1+alpha)
                    modify[modify>0] *= 2

                modify = (normalize(modify) + 2)/4 * 255
                modify = asInt8(modify)
                #modify = cv2.cvtColor(modify, cv2.COLOR_BGR2GRAY)
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #tmp = frame.astype(float)
                #thr = 160
                #tmp[modify>thr] -= modify[modify>thr]*0.2

                #modify[modify>=thr]=thr
                #modify[modify<thr]=0
                #frame = np.concatenate([asInt8(tmp), modify, frame], axis=1)
                frame = np.concatenate([modify, frame], axis=1)
                cv2.imshow('x', frame)
                key = cv2.waitKey(30)
                if key == ord('q'):
                    return 1
        return -1


if __name__ == '__main__':
    play = VideoPlay(0, depth=4, size=5)
    L = play.play()
