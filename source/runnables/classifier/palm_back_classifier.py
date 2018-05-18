import sys
import os

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..", "..")))

from data.datasets.palm_back_classifier.pb_classifier_ds_management import *
import data.regularization.regularizer as reg
import numpy as np
import data.datasets.crop.utils as u


# palm visible (+1.0)
# back visible (0.0)

batch_size = 2
path = resources_path("palm_back_classification_dataset_gray")


def to_uint8(xarr):
    ris = []
    for x1 in xarr:
        ris.append(np.array(x1, dtype=np.uint8))
    return ris


def multiply(xarr, harr):
    ris = []
    for i in range(len(xarr)):
        ris.append(np.array(xarr[i] * harr[i]))
    return ris


def means_stds(xarr):
    m = []
    s = []
    for x1 in xarr:
        m.append(x1.mean())
        s.append(x1.std())
    return m, s


def mean(vals, y, test):
    ris = 0
    n = 0
    for i in range(len(vals)):
        if y[i] == test:
            ris += vals[i]
            n += 1
    return ris/n


if __name__ == '__main__':

    regularizer = reg.Regularizer()
    regularizer.rgb2gray()

    # create_dataset_w_heatmaps(savepath=path, im_regularizer=regularizer)

    x, y, c, h = read_dataset_h(path=path, minconf=0.999)
    ind = 40
    u.showimage(np.array(np.dstack((x[ind], x[ind], x[ind])), dtype=np.uint8))
    m, s = means_stds(x)


