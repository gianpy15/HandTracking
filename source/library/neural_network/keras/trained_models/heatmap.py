import os
import sys

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..", "..")))

import keras.backend as kb

from library.neural_network.keras.models.heatmap import *
from data.naming import *
from keras.applications.mobilenet import preprocess_input


def relu6(x):
  return kb.relu(x, max_value=6)


model_path = models_path('deployment', 'transfer_mobilenet.h5')


def heatmap():
    model = km.load_model(model_path, custom_objects={'relu6': relu6})
    return model
