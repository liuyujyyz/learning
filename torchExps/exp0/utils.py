import cv2
import imageio
from tqdm import tqdm

class GIFWriter:
    def __init__(self):
        self.images = []
        self.H = None
        self.W = None
        self.channel = None

    def save(self, filename, fps=30, loop=0):
        with imageio.get_writer(filename, mode='I', fps=fps, loop=loop) as writer:
            for image in tqdm(self.images):
                writer.append_data(image)

    def append(self, image):
        shape = image.shape
        if len(shape) == 2:
            shape = tuple(shape) + (1,)
        if len(self.images) == 0:
            self.W, self.H, self.channel = shape
        else:
            W, H, C = shape
            assert (W==self.W) and (H==self.H) and (C==self.channel)

        self.images.append(image)


