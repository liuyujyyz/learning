import numpy as np 
import cv2
import sys
import os

def convert_fhq(h, w, msg):
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

def edge_det(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img, dtype='int')
    re = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            tmp = np.abs(img[max(i-1,0):i+2, max(j-1,0):j+2] - img[i,j])
            re[i][j] = tmp.max()
    return re

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

def main(argv):
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
    main(sys.argv)
