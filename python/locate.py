import cv2
import numpy as np
from decorators import timer
from cluster import Kmeans
from tqdm import tqdm

def extract(img):
    img = cv2.resize(img, (16, 16))
    U, S, V = np.linalg.svd(img)
    return V[0] 


@timer
def divide(img, window, stride=1):
    h, w, _ = img.shape
    parth, partw = window 
    out = []
    outImg = []
    steph = (h-parth)//stride
    stepw = (w-partw)//stride
    boxes = []
    for i in range(steph):
        for j in range(stepw):
            tmpImg = img[stride*i:stride*i+parth,stride*j:stride*j+partw]
            U = np.concatenate([extract(tmpImg[:,:,0]),extract(tmpImg[:,:,1]),extract(tmpImg[:,:,2])], axis=0)
            #U = extract(tmpImg[:,:,0])+extract(tmpImg[:,:,1])+extract(tmpImg[:,:,2])
            out.append(U)
            outImg.append(tmpImg)
            boxes.append((stride*i, stride*j, stride*i+parth, stride*j+partw))
    out = np.array(out)
    outImg = np.array(outImg)
    boxes = np.array(boxes)
    return out, outImg, boxes


def get_rep(filename, ID):
    img = cv2.imread(filename)
    rep, imgset, boxes = divide(img, (45, 45), 10)
    rep2, imgset2, boxes2 = divide(img, (90, 90), 20)
    rep3, imgset3, boxes3 = divide(img, (30, 30), 10)
    rep4, imgset4, boxes4 = divide(img, (60, 60), 20)

    rep = np.concatenate([rep, rep2, rep3, rep4], axis=0)
    boxes = np.concatenate([boxes, boxes2, boxes3, boxes4], axis=0)
    fileIndex = ID*np.ones((rep.shape[0],), dtype='int')
    return img, rep, boxes, fileIndex 


def findBackground(cato, index):
    return (cato.sum()*2 < cato.shape[0])


if __name__ == '__main__':
    reps = []
    imgsets = []
    boxess = []
    fileIndexs = []
    imgs = []
    dists = []
    numImg = 10 
    for i in tqdm(range(numImg)):
        img, rep, boxes, fileIndex = get_rep('../data/cat/2/pic%s.jpg'%(i), i)
        imgs.append(img)

        for j in range(3):
            cato, dist = Kmeans(rep, 2)

            if cato.sum() == 0:
                from IPython import embed
                embed()
            tag = int(cato.sum()*2 < cato.shape[0])
            if j > 0:
                tag = 1 - tag
            idx = np.where(cato == tag)[0]
            rep = rep[idx]
            boxes = boxes[idx]
            fileIndex = fileIndex[idx] 
            dist = dist[idx]
        
        reps.append(rep)
        boxess.append(boxes)
        fileIndexs.append(fileIndex)
        dists.append(dist)

    rep = np.concatenate(reps, axis=0)
    boxes = np.concatenate(boxess, axis=0)
    fileIndex = np.concatenate(fileIndexs, axis=0)
    dist = np.concatenate(dists, axis=0)
    
    while True:
        if rep.shape[0] < 10 * numImg:
            break
        cato, dist = Kmeans(rep, 2)
        tag = findBackground(cato, fileIndex)
        tag = 1 - tag
        print(set(cato), tag)

        idx = np.where(cato == tag)[0]
        nrep = rep[idx]
        nbox = boxes[idx]
        nfile = fileIndex[idx] 
        ndist = dist[idx]
        count = [(nfile==i).sum() for i in range(numImg)]
        
        if min(count) > 0:
            rep = nrep
            boxes = nbox
            fileIndex = nfile
            dist =ndist
        else:
            print(count)
            break

    maxi = dist.max()
    mini = dist.min()
    mean = dist.mean()
    ratio = 255 * (dist - mini) / (mean - mini)
    for i in range(rep.shape[0]):
        if dist[i] > mean:
            continue
        cv2.rectangle(imgs[fileIndex[i]], (boxes[i][1], boxes[i][0]), (boxes[i][3], boxes[i][2]), (int(ratio[i]),0,0), 1)
    for i in range(numImg):
        cv2.imshow('x', imgs[i])
        cv2.waitKey(0)
