#python2
import os, sys
import pickle, json
import numpy as np
import cv2
import argparse
pnt = []
tmp = []
image = None
N = None

def click_and_loc(event, x, y, flags, param):
    global pnt, tmp
    if event == cv2.EVENT_LBUTTONUP:
        tmp += [(x,y)]
        cv2.circle(image, tmp[-1], 2, (0,0,255), -1)
        cv2.imshow('image', image)
        if len(tmp) == N:
            pnt.append(tmp)
            tmp = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-of-ld', type=int, default=3)
    parser.add_argument('-t', '--target')
    args = parser.parse_args()
    N = args.num_of_ld
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', click_and_loc)

    result = {}
    L = list(os.walk(args.target))[0]
    p,d,f = L
    for item in f:
        img = cv2.imread(os.path.join(p,item))
        if img is None:
            continue
        image = img.copy()
        if img is None:
            continue
        pnt = []
        cv2.imshow('image', img)
        cv2.waitKey(0)
        print(pnt)
        result[os.path.join(p,item)] = pnt
    
    path = args.target.split('/')
    name = '_'.join(path)
    json.dump(result, open(os.path.join(args.target,'landmark.json'),'w'))
    cv2.destroyAllWindows()
        

