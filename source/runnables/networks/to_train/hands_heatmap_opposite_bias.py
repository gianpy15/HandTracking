import os
import sys

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..", "..")))

from library.neural_network.keras.models.heatmap import *
from library.neural_network.keras.custom_layers.heatmap_loss import *
from library.neural_network.tensorboard_interface.tensorboard_manager import TensorBoardManager as TBManager
from keras.engine import training as kt
import keras.models as km
from skimage.transform import rescale
from data.datasets.data_loader import load_dataset
from data.naming import *
from library.neural_network.keras.models.opposite_bias_model import opposite_bias_adversarial, opposite_bias_regularizer
from library.neural_network.keras.training.model_trainer import train_model
import keras.regularizers as kr
from data.augmentation.data_augmenter import Augmenter
from data.regularization.regularizer import Regularizer
import numpy as np

dataset_path = resources_path(os.path.join("hands_bounding_dataset", "network_test"))
m1_path = cropper_h5_path("cropper_opposite_bias_v1_m1")
m2_path = cropper_h5_path("cropper_opposite_bias_v1_m1")
m3_path = cropper_h5_path("cropper_opposite_bias_v1_m1")

TBManager.set_path("heat_maps")
train = True

# Hyper parameters
weight_decay = kr.l2(1e-5)
learning_rate = 1e-3
loss_delta = 0.8


# Load data
dataset = load_dataset(train_samples=3000,
                       valid_samples=1000,
                       use_depth=False)

# Augment data
log("Augmenting data...")
augmenter = Augmenter()
augmenter.shift_hue(prob=0.2, var=0.1).shift_sat(prob=0.2).shift_val(prob=0.2)
dataset[TRAIN_IN] = augmenter.apply_on_batch(dataset[TRAIN_IN])
dataset[VALID_IN] = augmenter.apply_on_batch(dataset[VALID_IN])
log("Augmentation end")

# Regularize data
log("Regularizing data...")
regularizer = Regularizer()
regularizer.normalize()
dataset[TRAIN_IN] = regularizer.apply_on_batch(dataset[TRAIN_IN])
dataset[VALID_IN] = regularizer.apply_on_batch(dataset[VALID_IN])
log("Regularization end")


def attach_heat_map(inputs, fitted_model_positive_path, fitted_model_negative_path):
    _inputs = inputs[:, :, :, 0:3]
    fitted_model_positive = km.load_model(fitted_model_positive_path)
    fitted_model_negative = km.load_model(fitted_model_negative_path)
    outputs_positive = fitted_model_positive.predict(inputs)
    outputs_negative = fitted_model_negative.predict(inputs)
    rescaled_positive = []
    rescaled_negative = []
    for img in outputs_positive:
        rescaled_positive.append(rescale(img, 4.0))
    for img in outputs_negative:
        rescaled_negative.append(rescale(img, 4.0))

    outputs_positive = np.array(rescaled_positive)
    outputs_negative = np.array(rescaled_negative)
    inputs_ = np.concatenate((_inputs, outputs_positive, outputs_negative), axis=-1)
    print(np.shape(inputs_))
    return inputs_


# Build up the model
# Model with high penalty for P(x = 1 | not hand)
model1 = train_model(dataset=dataset,
                     model_generator=lambda: opposite_bias_adversarial(weight_decay=weight_decay),
                     loss=lambda x, y: prop_heatmap_penalized_fp_loss(x, y, -1.85, loss_delta),
                     learning_rate=learning_rate,
                     patience=5,
                     tb_path="heat_maps/opposite_bias_v1_m1",
                     model_name="cropper_opposite_bias_v1_m1",
                     model_type=CROPPER,
                     batch_size=30,
                     epochs=50,
                     verbose=False)

# Model with high penalty for P(x = 0 | hand)
model2 = train_model(dataset=dataset,
                     model_generator=lambda: opposite_bias_adversarial(weight_decay=weight_decay),
                     loss=lambda x, y: prop_heatmap_penalized_fn_loss(x, y, -1.85, loss_delta),
                     learning_rate=learning_rate,
                     patience=5,
                     tb_path="heat_maps/opposite_bias_v1_m2",
                     model_name="cropper_opposite_bias_v1_m2",
                     model_type=CROPPER,
                     batch_size=30,
                     epochs=50,
                     verbose=False)

dataset[TRAIN_IN] = attach_heat_map(dataset[TRAIN_IN], m1_path, m2_path)
dataset[VALID_IN] = attach_heat_map(dataset[VALID_IN], m1_path, m2_path)

"""
# Third Model
model3 = train_model(dataset=dataset,
                     model_generator=lambda: opposite_bias_regularizer(weight_decay=weight_decay),
                     learning_rate=learning_rate,
                     patience=5,
                     tb_path="heat_maps/opposite_bias_v1_m3",
                     model_name="cropper_opposite_bias_v1_m3",
                     model_type=CROPPER,
                     batch_size=20,
                     epochs=1,
                     verbose=False)
"""
