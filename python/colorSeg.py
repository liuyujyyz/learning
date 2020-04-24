import cv2
from cluster import Kmeans, get_center 
import numpy as np

def colorSeg(img, K = 6):
    h, w, c = img.shape 
    tmpimg = cv2.resize(img, (w//4, h//4))
    data = tmpimg.reshape((tmpimg.shape[0]*tmpimg.shape[1], 3))
    N = data.shape[0]
    cato, dist = Kmeans(data, K)
    center = get_center(data, cato, K)
    palette = np.zeros((K, h, w, 3))
    for i in range(K):
        palette[i] = center[i]
    palette = palette.astype('uint8')

    outimg = []
    ratios = []
    ratio_sum = np.zeros((h, w))
    for i in range(K):
        dist = -((img[:,:] - center[i])**2).sum(axis=2)
        ratio = np.exp(dist-dist.max())
        ratio = ratio / ratio.sum() 
        ratios.append(ratio)
        ratio_sum += ratio

    print(ratio_sum.max(), ratio_sum.min())
    total = np.zeros((h, w, c))
    for i in range(K):
        ratio = ratios[i]
        ratio[ratio_sum > 0] /= ratio_sum[ratio_sum>0]
        ratio[ratio_sum < 1e-8] = 1/K 
        ratios[i] = ratio
        ratio = ratio.reshape(ratio.shape + (1,))
        tmpimg = (img * ratio).astype('uint8')
        outimg.append(tmpimg)
        total += tmpimg
    total = total.astype('uint8')
    cv2.imshow('x', total)
    cv2.waitKey(0)
    from IPython import embed
    embed()
    return outimg

if __name__ == '__main__':
    img = cv2.imread('/home/liuyu/Pictures/kinship/VivoTwins/COMPARE_FAILED/Real-07-04-22_22_33.362-C-87-L-0-B-0-M-0.nv21.640.480.png')
    cv2.imshow('ori', img)
    cv2.waitKey(0)
    out = colorSeg(img, K=6)
    for item in out:
        cv2.imshow('x%s'%id(item), item)
        cv2.waitKey(0)

