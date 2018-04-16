import numpy as np
from image_manipulation.hsv import rgb2hsv, hsv2rgb

HUE = 0
SAT = 1
VAL = 2


class Augmenter:
    def __init__(self):
        self.prob = [0.0, 0.0, 0.0]
        self.var = [1, 1, 1]
        self.append = False

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
        if not self.append:
            for idx in range(len(batch)):
                self.apply(batch[idx])
            return batch

        ris = list(batch)
        for elem in batch:
            new_elem = np.array(elem)
            if self.apply(new_elem):
                ris.append(new_elem)
        return np.array(ris)

    def append(self, flag=True):
        self.append = flag

    def __shift(self, comp, prob=1.0, var=1):
        self.prob[comp] = prob
        self.var[comp] = var

    def shift_hue(self, prob=1.0, var=1):
        self.__shift(HUE, prob, var)

    def shift_sat(self, prob=1.0, var=1):
        self.__shift(SAT, prob, var)

    def shift_val(self, prob=1.0, var=1):
        self.__shift(VAL, prob, var)


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
    from image_loader.image_loader import load
    from data_manager.path_manager import resources_path
    from matplotlib import pyplot as mplt
    from neural_network.keras.utils.data_loader import load_crop_dataset
    from neural_network.keras.utils.naming import *

    def showimg(img):
        mplt.imshow(img)
        mplt.show()


    dataset = load_crop_dataset(10, 10)
    augmenter = Augmenter()
    augmenter.shift_hue(0.5, 2)
    augmenter.shift_sat(0.2, 4)
    augmenter.shift_val(0.3, 1)
    dataset[TRAIN_IN] = augmenter.apply_on_batch(dataset[TRAIN_IN])
    print(np.shape(dataset[TRAIN_IN]))

    def speedtest():
        hsv2rgb(img)
        # component_shift(img, shamt=0.2)


    # import timeit

    for img in dataset[TRAIN_IN]:
        showimg(img)
    # print(timeit.timeit(speedtest, number=100))
