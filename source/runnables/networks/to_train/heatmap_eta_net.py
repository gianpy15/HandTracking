import os
import sys

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..", "..")))

from library.neural_network.keras.custom_layers.heatmap_loss import *
from library.neural_network.tensorboard_interface.tensorboard_manager import TensorBoardManager as TBManager
from data.datasets.data_loader import load_dataset
from data.naming import *
import keras as K
from library.neural_network.keras.models.eta_net import eta_net
from library.neural_network.keras.training.model_trainer import train_model
import keras.regularizers as kr
from data.augmentation.data_augmenter import Augmenter
from data.regularization.regularizer import Regularizer
import numpy as np


if __name__ == '__main__':
    model = 'cropper_eta_net_v1'

    m1_path = cropper_h5_path(model)

    TBManager.set_path("heat_maps")
    train = True

    # Hyper parameters
    weight_decay = kr.l2(1e-5)
    learning_rate = 1e-4

    # Load data
    dataset = load_dataset(train_samples=500,
                           valid_samples=100,
                           use_depth=False)

    # Augment data
    log("Augmenting data...")
    augmenter = Augmenter()
    augmenter.shift_hue(prob=0.25).shift_sat(prob=0.25).shift_val(prob=0.25)
    dataset[TRAIN_IN] = augmenter.apply_on_batch(dataset[TRAIN_IN])
    dataset[VALID_IN] = augmenter.apply_on_batch(dataset[VALID_IN])
    log("Augmentation end")

    input_shape = np.shape(dataset[TRAIN_IN][0])
    print("Input shape: {}".format(input_shape))

    # Regularize data
    log("Regularizing data...")
    regularizer = Regularizer()
    regularizer.normalize()
    dataset[TRAIN_IN] = regularizer.apply_on_batch(dataset[TRAIN_IN])
    dataset[VALID_IN] = regularizer.apply_on_batch(dataset[VALID_IN])
    log("Regularization end")
    reg2 = Regularizer()
    reg2.fixresize(height=60, width=80)
    dataset[TRAIN_TARGET] = reg2.apply_on_batch(dataset[TRAIN_TARGET])
    dataset[VALID_TARGET] = reg2.apply_on_batch(dataset[VALID_TARGET])
    input_shape = np.shape(dataset[TRAIN_IN][0])
    target_shape = np.shape(dataset[TRAIN_TARGET][0])
    print("Input shape: {}".format(input_shape))
    print("Target shape: {}".format(target_shape))

    # Build up the model
    # Model with high penalty for P(x = 1 | not hand)
    model1 = train_model(dataset=dataset,
                         model_generator=lambda: eta_net(input_shape=input_shape, weight_decay=weight_decay,
                                                         dropout_rate=0.5,
                                                         activation=lambda: K.layers.LeakyReLU(alpha=0.1)),
                         loss=lambda x, y: prop_heatmap_penalized_fp_loss(x, y, -1.85, 3),
                         learning_rate=learning_rate,
                         patience=5,
                         tb_path="heat_maps/" + model,
                         model_name=model,
                         model_type=CROPPER,
                         batch_size=10,
                         epochs=100,
                         verbose=True)
