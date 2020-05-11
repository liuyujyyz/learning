from constants import *
from utils import *
from particle import Particle, force, forceUnion
import numpy as np
import cv2
from tqdm import tqdm

units = [
        Particle((0, 0, 0), (0, 500, 0), 5.965e24, 0), 
        Particle((4.84e8, 0, 0), (0, 800, 0), 7.349e23, 0),
        Particle((7e8, 2e8, 0), (0, -500, 0), 5.965e24, 0), 
        Particle((-4.84e8, 0, 0), (0, 800, 0), 7.349e25, 0),
        ] 

Len = 4

size = 500
fps = 30
ruler = 1e10
for j in range(Len):
    print(units[j])

#gif = GIFWriter()
video = cv2.VideoWriter('test2.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (size, size))

for i in tqdm(range(500*fps*30+1)):
    forces = []
    for j in range(Len):
        f = forceUnion(units[j], units[:j] + units[j+1:Len])
        forces.append(f)

    for j in range(Len):
        units[j].move(forces[j], 60) 
    
    if i % 500 == 0:
        img = np.zeros((size,size,3), dtype='uint8')
        for j in range(Len):
            x = int((units[j].position.value[0]/ruler+1)*size/2)
            y = int((units[j].position.value[1]/ruler+1)*size/2)
            cv2.circle(img, (x, y), 5, (128, 64*(j//2), 64*(j%2)), -1)
#        gif.append(img)
        video.write(img)

video.release() 
#gif.save('test.gif', fps=fps)

