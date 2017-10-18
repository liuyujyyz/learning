import numpy as np
import cv2
# code for align face to mean_face using 5 points;
# width/height remains if freeze is True; otherwise it varies.
  
# average landmarks
cfg = {}
cfg['left_eye'] = [-0.17, -0.17]
cfg['right_eye'] = [0.17, -0.17]
cfg['nose'] = [-0.0, 0.00]
cfg['left_mouth'] = [-0.14, 0.20]
cfg['right_mouth'] = [0.14, 0.20]
cfg['mean_face'] = [cfg['left_eye'], cfg['right_eye'], cfg['nose'], cfg['left_mouth'], cfg['right_mouth']]
 
def align_5p(img, lm, freeze):
    width = 256
    mf = np.array(cfg['mean_face'])
     
    # Assumptions:
    # 1. The output image size is 256x256 pixels
    # 2. The distance between two eye pupils is 70 pixels
    ratio = 70.0 / (256.0 * 0.35) # magic number 0.34967 to compensate scaling from average landmarks
     
    # In an aligned face image, the ratio between the vertical distances from eye to the top and bottom is 1:1.42
    ratioy = (cfg['left_eye'][1] * ratio + 0.5) * (1 + 1.42)
    mf[:,0] = (mf[:,0] * ratio + 0.5) * width
    mf[:,1] = (mf[:,1] * ratio + 0.5) * width / ratioy
    mx = mf[:,0].mean()
    my = mf[:,1].mean()
    dmx = lm[:,0].mean()
    dmy = lm[:,1].mean()
    mat = np.zeros((3,3), dtype=float)
    ux = mf[:,0] - mx
    uy = mf[:,1] - my
    dux = lm[:,0] - dmx
    duy = lm[:,1] - dmy
    c1 = (ux * dux + uy * duy).sum()
    c2 = (ux * duy - uy * dux).sum()
    c3 = (dux **2 + duy **2).sum()
    a = c1 / c3
    b = c2 / c3
 
    if freeze:
        kx = 1
        ky = 1
    else:
        kx = (ux*(a*dux + b*duy)).sum()/((a*dux + b*duy)**2).sum()
        ky = (uy*(a*duy - b*dux)).sum()/((a*duy - b*dux)**2).sum()
     
    s = c3/(c1 **2 + c2 **2)
    ka = c1 * s
    kb = c2 * s
 
    transform = np.zeros((2,3))
    transform[0][0] = kx*a
    transform[0][1] = kx*b
    transform[0][2] = mx - kx * a * dmx - kx * b * dmy
    transform[1][0] = -ky*b
    transform[1][1] = ky*a
    transform[1][2] = my - ky * a * dmy + ky * b * dmx
    jmg = cv2.warpAffine(img, transform[:2], (width, width))
    return jmg,transform
 
def alignimg(img, ld, freeze=True):
    lds = [ld['LEFTEYE_PUPIL'], ld['RIGHTEYE_PUPIL'], ld['NOSE_TIP'], ld['MOUTH_LEFTCORNER'], ld['MOUTH_RIGHTCORNER']]
    lds = np.array(lds)
    img2, mat = align_5p(img, lds, freeze)
    ld2 = {}
    for i in ld:
        ld2[i] = np.array(np.dot(mat[:2], np.array(ld[i]+[1]).reshape((3,1)))).reshape(-1).tolist()
    return img2, ld2
