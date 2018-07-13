import os
import sys

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..", "..")))

from keras.applications.mobilenet import relu6, preprocess_input
import data.datasets.crop.utils as u
from data.datasets.io.image_loader import load
from library.neural_network.keras.models.heatmap import *
from data.naming import *
from library.utils.visualization_utils import get_image_with_mask
import numpy as np
from library.utils.hsv import rgb2hsv, hsv2rgb

if __name__ == '__main__':
    # pls specify the name of the image, (png, jpg)
    image_name = "hands.png"

    dataset_path = resources_path("hands_bounding_dataset", "network_test")
    png_path = resources_path("gui", image_name)
    model_path = models_path('deployment', 'transfer_mobilenet.h5')
    read_from_png = True
    preprocessing = True

    height = 224
    width = 224
    if read_from_png:
        images = load(png_path, force_format=(height, width, 3))
        print("Saturation mean: {}".format(np.mean(rgb2hsv(images[0])[:, :, 1])))
        print("Value mean: {}".format(np.mean(images[0, :, :, 2])))
        print("Hue mean: {}".format(np.mean(images[0, :, :, 0])))
        hsv2rgb(images[0])
        if preprocessing:
            images_ = preprocess_input(images * 255)

    # Build up the model
    model = km.load_model(model_path, custom_objects={'relu6': relu6})

    # Testing the model getting some outputs
    net_out = model.predict(images_ if preprocessing else images)
    imgs = get_image_with_mask(images, net_out)
    u.showimage(imgs[0]/255)
