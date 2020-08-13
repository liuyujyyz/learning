import cv2
import numpy as np 
import os
from utils.libs.imgproc import conv2d, normalize, as_img, lighter, formula_size, gamma_lighter, lighter_yuv

np.random.seed(0)


def align():
    L = os.walk('/home/liuyu/liuyu_lib/learning/data/cat/')
    data = []
    label = []
    for p, d, f in L:
        for item in f:
            img = cv2.imread(os.path.join(p, item))
            if img is not None:
                img = formula_size(img, shorter=64)
                data.append(img)
                label.append(p)
    from IPython import embed
    embed()
                


def main():
    p, d, f = next(os.walk('/home/liuyu/Desktop/test_images'))
    img_gallary = []
    for item in f:
        img = cv2.imread(os.path.join(p, item))
        if img is not None:
            img_gallary.append(formula_size(img))

    for item in img_gallary:
        imgA = lighter(item)
        imgB = gamma_lighter(item, avg=60)
        imgC = lighter(imgB)
        imgD = lighter(item, channel_wise=False)
        imgE = lighter_yuv(item)
        img0 = np.concatenate([item, imgE], axis=0)
        img1 = np.concatenate([imgA, imgB], axis=0)
        img2 = np.concatenate([imgC, imgD], axis=0)
        img = np.concatenate([img0, img1, img2], axis=1)
        cv2.imshow('x', img)
        key = cv2.waitKey(0)

        img = as_img(conv2d(img, np.ones((3,3,3,3))/27))
        cv2.imshow('y', img)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
