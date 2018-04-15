import numpy as np
import random as rand
from image_manipulation.hsv import rgb2hsv, hsv2rgb
from numba import jit, prange

HUE = 0
SAT = 1
VAL = 2


class Augmenter:
    def __init__(self):
        self.ops = []

    def apply(self, frame: np.ndarray):
        for op in self.ops:
            pass
        return frame

    def apply_on_batch(self, batch: np.ndarray):
        ris = []
        for elem in batch:
            ris.append(self.apply(elem))
        return np.array(ris)


def truncated_gauss_random(var):
    r = np.random.normal(scale=var)
    return np.modf(r)[0]


def component_shift(img: np.ndarray, shamt: float, comp=0, rotate=True):
    img[:, :, comp] += shamt
    if rotate:
        np.modf(img[:, :, comp], out=img[:, :, comp])
    elif shamt > 0:
        img[np.greater(img, 1)] = 1
    else:
        img[np.greater(0, img)] = 0
    return img


if __name__ == '__main__':
    from image_loader.image_loader import load
    from data_manager.path_manager import resources_path
    from matplotlib import pyplot as mplt

    def showimg(img):
        mplt.imshow(img)
        mplt.show()
    img = load(resources_path("gui", "hands.png"))[0]
    showimg(img)
    rgb2hsv(img)
    for shamt in np.arange(-1, 1, 0.4):
        cp = np.array(img)
        out = component_shift(cp, shamt=shamt, comp=1, rotate=False)
        showimg(hsv2rgb(out))
    img = np.array(img, dtype=np.float32)

    def speedtest():
        hsv2rgb(img)
        # component_shift(img, shamt=0.2)

    import timeit
    print(img.shape)
    print(timeit.timeit(speedtest, number=100))
