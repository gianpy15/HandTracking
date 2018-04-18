import os
import sys

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..", "..")))

from library.neural_network.keras.models.heatmap import *
from library.neural_network.keras.custom_layers.heatmap_loss import *
from library.neural_network.tensorboard_interface.tensorboard_manager import TensorBoardManager as TBManager
from keras.engine import training as kt
from skimage.transform import rescale
from data.datasets.data_loader import load_dataset
from data.naming import *
from library.neural_network.keras.training.model_trainer import train_model
import keras.regularizers as kr
from data.augmentation.data_augmenter import Augmenter
from data.regularization.regularizer import Regularizer
import numpy as np

dataset_path = resources_path(os.path.join("hands_bounding_dataset", "network_test"))
m1_path = croppers_path("cropper_v6_m1.h5")

TBManager.set_path("heat_maps")
train = True

# Hyper parameters
weight_decay = kr.l2(1e-5)
learning_rate = 1e-3



# Load data
dataset = load_dataset(train_samples=3000,
                       valid_samples=100,
                       use_depth=False)
# Augment data
print("Augmenting data...")
augmenter = Augmenter()
augmenter.shift_hue(prob=0.25).shift_sat(prob=0.25).shift_val(prob=0.25)
dataset[TRAIN_IN] = augmenter.apply_on_batch(dataset[TRAIN_IN])
dataset[VALID_IN] = augmenter.apply_on_batch(dataset[VALID_IN])
print("Augmentation end")

# Regularize data
print("Regularizing data...")
regularizer = Regularizer()
regularizer.normalize()
dataset[TRAIN_IN] = regularizer.apply_on_batch(dataset[TRAIN_IN])
dataset[VALID_IN] = regularizer.apply_on_batch(dataset[VALID_IN])
print("Regularization end")


def attach_heat_map(inputs, fitted_model: kt.Model):
    _inputs = inputs[:, :, :, 0:3]
    outputs = fitted_model.predict(inputs)
    rescaled = []
    for img in outputs:
        rescaled.append(rescale(img, 4.0))
    outputs = np.array(rescaled)
    inputs_ = np.concatenate((_inputs, outputs), axis=-1)
    return inputs_


# Build up the model
# First model part
dataset[TRAIN_IN] = np.concatenate((dataset[TRAIN_IN], np.zeros(shape=np.shape(dataset[TRAIN_IN])[0:-1] + (1,))),
                                   axis=-1)
dataset[VALID_IN] = np.concatenate((dataset[VALID_IN], np.zeros(shape=np.shape(dataset[VALID_IN])[0:-1] + (1,))),
                                   axis=-1)
model1 = train_model(dataset=dataset,
                     model_generator=lambda: incremental_predictor_1(weight_decay=weight_decay),
                     learning_rate=learning_rate,
                     patience=5,
                     tb_path="heat_maps/m1",
                     model_name="cropper_v6_m1",
                     model_type=CROPPER,
                     batch_size=20,
                     epochs=50,
                     loss_white_prio=-1.85,
                     verbose=True)

dataset[TRAIN_IN] = attach_heat_map(dataset[TRAIN_IN], model1)
dataset[VALID_IN] = attach_heat_map(dataset[VALID_IN], model1)
# Second Model
model2 = train_model(dataset=dataset,
                     model_generator=lambda: incremental_predictor_1(weight_decay=weight_decay),
                     learning_rate=learning_rate,
                     patience=5,
                     tb_path="heat_maps/m2",
                     model_name="cropper_v6_m2",
                     model_type=CROPPER,
                     batch_size=20,
                     epochs=50,
                     loss_white_prio=-1.5,
                     verbose=True)

dataset[TRAIN_IN] = attach_heat_map(dataset[TRAIN_IN], model2)
dataset[VALID_IN] = attach_heat_map(dataset[VALID_IN], model2)
# Third Model
model3 = train_model(dataset=dataset,
                     model_generator=lambda: incremental_predictor_1(weight_decay=weight_decay),
                     learning_rate=learning_rate,
                     patience=5,
                     tb_path="heat_maps/m3",
                     model_name="cropper_v6_m3",
                     model_type=CROPPER,
                     batch_size=20,
                     epochs=50,
                     verbose=False)
