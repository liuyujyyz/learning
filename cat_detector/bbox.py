#python2
import os, sys
import pickle
import numpy as np
import cv2

pnt = []
tmp = None
image = None

def click_and_loc(event, x, y, flags, param):
    global pnt, tmp
    if event == cv2.EVENT_LBUTTONDOWN:
        tmp = [(x,y)]
    elif event == cv2.EVENT_LBUTTONUP:
        tmp += [(x,y)]
        pnt.append(tmp)
        cv2.rectangle(image, tmp[0], tmp[1], (0,0,255), 2)
        cv2.imshow('image', image)

if __name__ == '__main__':
    target = sys.argv[-1]
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', click_and_loc)

    result = {}
    L = list(os.walk(target))[0]
    p,d,f = L
    for item in f:
        img = cv2.imread(os.path.join(p,item))
        image = img.copy()
        if img is None:
            continue
        pnt = []
        cv2.imshow('image', img)
        cv2.waitKey(0)
        result[item] = pnt

    pickle.dump(result, open('%s.pkl'%(target.split('/')[-1]),'wb'))
    cv2.destroyAllWindows()
        

