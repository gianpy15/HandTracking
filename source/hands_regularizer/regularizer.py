import numpy as np
from scipy.misc import imresize
import scipy
import hands_bounding_utils.utils as u

PARAMS = 'PARAMS' #
NORMALIZE_AVG_VARIANCE = 'NORM_A_V'
RGB2GREY = 'RGB2GREY' #
RESIZEPERC = 'RESIZEPERC' #
RESIZEFIX = 'RESIZEFIX' #
PADDING = 'PADDING' #
OPS = 'OPS' #

class RegularizerParametersCreator:
    def __init__(self):
        self.pars = dict()
        self.pars[OPS] = []

    def padding(self, right_pad, left_pad):
        self.pars[OPS].append(PADDING)
        self.pars[PADDING] = [right_pad, left_pad]

    def rgb2gray(self):
        self.pars[OPS].append(RGB2GREY)

    def percresize(self, perc):
        self.pars[OPS].append(RESIZEPERC)
        self.pars[RESIZEPERC] = perc

    def fixresize(self, height, width):
        self.pars[OPS].append(RESIZEFIX)
        self.pars[RESIZEFIX] = [height, width]


def __add_padding(image, right_pad, bottom_pad):
    image = np.hstack((image, np.zeros([image.shape[0], right_pad, image.shape[2]], dtype=image.dtype)))
    image = np.vstack((image, np.zeros([bottom_pad, image.shape[1], image.shape[2]], dtype=image.dtype)))
    return image


def __rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.reshape([gray.shape[0], gray.shape[1], 1])


def __imresizeperc(frame, rate):
    return imresize(frame, rate)


def __fixed_resize(frame, size):
    return imresize(frame, size)


test = scipy.misc.imread("t.jpg")
test = np.array(test)

res = __fixed_resize(test, (250, 200))
u.showimage(res)