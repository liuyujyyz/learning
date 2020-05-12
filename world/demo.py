from constants import *
from utils import *
from particle import Particle, force, forceUnion
import numpy as np
import cv2
from tqdm import tqdm

units = [
        Particle((0, 0, 0), (np.random.randint(-800,800), np.random.randint(-8000,8000), 0), 5.965e24, 0), 
        Particle((4.84e8, 0, 0), (np.random.randint(-800,800), np.random.randint(-8000,8000), 0), 7.349e23, 0),
        Particle((7e8, 2e8, 0), (np.random.randint(-800,0), np.random.randint(-8000,8000), 0), 5.965e24, 0), 
        Particle((-4.84e8, 0, 0), (np.random.randint(0,800), np.random.randint(-8000,8000), 0), 7.349e22, 0),
        ] 

Len = 4

size = 500
fps = 30
ruler = 1e10
for j in range(Len):
    print(units[j])

#gif = GIFWriter()
#video = cv2.VideoWriter('test2.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (size, size))
#for i in tqdm(range(500*fps*30+1)):
i = 0
biasX = 0
biasY = 0
duration = 5
while True:
#for i in tqdm(range(500*fps*3000+1)):
    forces = []
    for j in range(Len):
        f = forceUnion(units[j], units[:j] + units[j+1:Len])
        forces.append(f)

    for j in range(Len):
        units[j].move(forces[j], 60) 
    
    if i % duration == 0:
        img = np.zeros((size,size,3), dtype='uint8')
        for j in range(Len):
            x = int((units[j].position.value[0]/ruler+1)*size/2 + biasX)
            y = int((units[j].position.value[1]/ruler+1)*size/2 + biasY)
            cv2.circle(img, (x, y), 5, (128, 64*(j//2), 64*(j%2)), -1)
#        gif.append(img)
#        video.write(img)
        cv2.imshow('x', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('w'):
            biasY += 1
        if key == ord('s'):
            biasY -= 1
        if key == ord('a'):
            biasX += 1
        if key == ord('d'):
            biasX -= 1
        if key == ord('o'):
            ruler *= 2
        if key == ord('p'):
            ruler /= 2

#video.release() 
#gif.save('test.gif', fps=fps)

