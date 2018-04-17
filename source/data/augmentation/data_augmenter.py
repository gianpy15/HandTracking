import numpy as np
from library.utils.hsv import rgb2hsv, hsv2rgb

HUE = 0
SAT = 1
VAL = 2


class Augmenter:
    def __init__(self):
        self.prob = [.0, .0, .0]
        self.var = [.15, .15, .15]
        self.in_place = True

    def apply(self, frame: np.ndarray):
        flag = False
        for comp in [HUE, SAT, VAL]:
            if self.prob[comp] > 0:
                rand = np.random.random()
                if rand < self.prob[comp]:
                    if not flag:
                        rgb2hsv(frame)
                    component_shift(frame, truncated_gauss_random(self.var[comp]), comp)
                    flag = True
        if flag:
            hsv2rgb(frame)
        return flag

    def apply_on_batch(self, batch: np.ndarray):
        if self.in_place:
            for idx in range(len(batch)):
                self.apply(batch[idx])
            return batch

        ris = list(batch)
        for elem in batch:
            new_elem = np.array(elem)
            if self.apply(new_elem):
                ris.append(new_elem)
        return np.array(ris)

    def __shift(self, comp, prob=1.0, var=.15):
        self.prob[comp] = prob
        self.var[comp] = var

    def shift_hue(self, prob=1.0, var=.15):
        self.__shift(HUE, prob, var)
        return self

    def shift_sat(self, prob=1.0, var=.15):
        self.__shift(SAT, prob, var)
        return self

    def shift_val(self, prob=1.0, var=.15):
        self.__shift(VAL, prob, var)
        return self


def truncated_gauss_random(var):
    r = np.random.normal(scale=var)
    return np.modf(r, dtype=np.float32)[0]


def component_shift(img: np.ndarray, shamt: np.float32, comp=HUE):
    rotate = (comp == HUE)
    img[:, :, comp] += shamt
    if rotate:
        np.modf(img[:, :, comp], out=img[:, :, comp])
    elif shamt > 0:
        img[np.greater(img, 1)] = 1
    else:
        img[np.greater(0, img)] = 0
    return img


if __name__ == '__main__':
    from data.datasets.io.image_loader import load
    from matplotlib import pyplot as mplt
    from data.datasets.data_loader import load_crop_dataset
    from data.naming import *
    import timeit

    def showimg(img):
        mplt.imshow(img)
        mplt.show()


    dataset = load_crop_dataset(10, 0)
    augmenter = Augmenter()
    augmenter.shift_hue(0.5, .15)
    augmenter.shift_sat(1.0, .15)
    augmenter.shift_val(0.5, .15)
    augmenter.in_place = True
    augmenter.apply_on_batch(dataset[TRAIN_IN])

    for img in dataset[TRAIN_IN]:
        showimg(img)

    def speedtest():
        augmenter.apply_on_batch(dataset[TRAIN_IN])

    print(timeit.timeit(speedtest, number=100))

