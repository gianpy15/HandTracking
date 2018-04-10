import numpy as np
from scipy.misc import imresize
import scipy
import hands_bounding_utils.utils as u

PARAMS = 'PARAMS'
NORMALIZE_AVG_VARIANCE = 'NORM_A_V'
RGB2GREY = 'RGB2GREY'
RESIZEPERC = 'RESIZEPERC'
RESIZEFIX = 'RESIZEFIX'
PADDING = 'PADDING'
HEATMAPS_TH = 'HEATMAPS_TH'
OPS = 'OPS'


class Regularizer:
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

    def normalize(self):
        self.pars[OPS].append(NORMALIZE_AVG_VARIANCE)

    def heatmaps_threshold(self, thresh):
        self.pars[OPS].append(HEATMAPS_TH)
        self.pars[HEATMAPS_TH] = thresh

    def apply(self, frame):
        frame = np.array(frame)
        for op in self.pars[OPS]:
            if op == PADDING:
                params = self.pars[PADDING]
                frame = add_padding(frame, params[0], params[1])
                continue
            if op == RGB2GREY:
                frame = rgb2gray(frame)
                continue
            if op == RESIZEPERC:
                frame = imresizeperc(frame, self.pars[RESIZEPERC])
                continue
            if op == RESIZEFIX:
                frame = fixed_resize(frame, self.pars[RESIZEFIX])
                continue
            if op == HEATMAPS_TH:
                frame = heat_thresh(frame, self.pars[HEATMAPS_TH])
                continue
            frame = normalize(frame)
        return frame

    def apply_on_batch(self, batch):
        ris = np.empty(shape=np.shape(batch))
        for idx in range(len(batch)):
            ris[idx] = self.apply(batch[idx])
        return ris


def add_padding(image, right_pad, bottom_pad):
    image = np.hstack((image, np.zeros([image.shape[0], right_pad, image.shape[2]], dtype=image.dtype)))
    image = np.vstack((image, np.zeros([bottom_pad, image.shape[1], image.shape[2]], dtype=image.dtype)))
    return image


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.reshape([gray.shape[0], gray.shape[1], 1])


def imresizeperc(frame, rate):
    return imresize(frame, rate)


def fixed_resize(frame, size):
    return imresize(frame, size)


def normalize(frame):
    avg = frame.mean()
    std = frame.std()
    frame = (frame - avg) / std
    return frame


def heat_thresh(heat, thresh):
    heatm = np.zeros(np.shape(heat))
    heatm[heat > thresh] = 1
    return heatm


if __name__ == '__main__':
    test = scipy.misc.imread("t.jpg")
    r = Regularizer()
    r.padding(50, 50)
    r.fixresize(50, 50)
    res = r.apply(test)
    u.showimage(res)
