import numpy as np
import imageio
from tqdm import tqdm


def add_lines(img, width=10):
    pattern = np.array(range(img.shape[1]))
    pattern = np.cos(pattern / (width*4)*2*np.pi) * 20
    pattern = pattern.reshape((1, img.shape[1], ) + (1,)*(img.ndim-2))
    img = np.clip(img + pattern, 0, 255).astype('uint8')
    return img


def hist_to_image(hist, bins=512):
    H = len(hist)
    W = bins 
    hist = np.array(hist)
    scale = (hist.max() - hist.min())/(bins-1)
    hist = ((hist - hist.min()) / scale).astype('int')
    image = np.ones((W, H), dtype='uint8') * 255
    for i in range(H):
        image[-hist[i]:, i] = 128
    return image


class GIFWriter:
    def __init__(self):
        self.images = []
        self.H = None
        self.W = None
        self.channel = None

    def save(self, filename, fps=30, loop=0):
        with imageio.get_writer(filename, mode='I', fps=fps, loop=0) as writer:
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


if __name__ == '__main__':
    w = GIFWriter()
    w.save('cache/test.gif', fps=30, loop=1)

