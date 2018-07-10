import os
import sys

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..", "..")))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from sklearn import datasets
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import linear_model
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold

from sklearn.decomposition import PCA

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from data.naming import *
from keras.losses import mean_squared_error
from data.datasets.data_loader import *
from data.datasets.jlocator.heatmaps_to_hand import heatmaps_to_hand
from library.geometry.formatting import *
from library.neural_network.keras.custom_layers.heatmap_loss import *
from data import *
from library import *
import keras as K
from library.neural_network.keras.models.transfer_learning import transfer_mobile_net_joints
from library.neural_network.keras.training.model_trainer import train_model
from library.neural_network.batch_processing.processing_plan import ProcessingPlan
from library.utils.visualization_utils import joint_skeleton_impression
from keras.applications.mobilenet import preprocess_input as preprocess_mobile
from library.neural_network.keras.models.joints_regressor import regressor

# ####################### HYPERPARAMETERS #######################

# NETWORK NAME (used for naming saved files)
# play with this adding extentions when saving models with different hyperparameter configurations
model = 'regressor_jlocator'

# TRAINING PARAMETERS:

# the number of training samples loaded
train_samples = 2  # >=1

# the number of validation samples loaded
valid_samples = 2  # >=1

# the number of samples used for each batch
# higher batch size leads to more significant gradient (less variance in gradient)
# but a batch size too high may not fit into the graphics video memory.
batch_size = 2  # >=1

# the number of epochs to perform without improvements in validation accuracy before triggering early stopping
# higher patience allows bridging greater "hills" but with obvious downsides in case the overfitting hill never ends
patience = 1000  # >=1

# the maximum number of epochs to perform before stopping.
# notice: usually this term is not influential because early stopping triggers first
epochs = 10  # >=1

# learning rate used for optimization
# higher learning rates lead to definitely faster convergence but possibly unstable behaviour
# setting a lower learning rate may also allow reaching smaller and deeper minima in the loss
# use it to save training time, but don't abuse it as you may lose the best solutions
learning_rate = 1e-4  # >0

# NETWORK PARAMETERS

# the dropout rate to be used in the entire network
# dropout will make training and learning more difficult as it shuts down random units at training time
# but it will improve generalization a lot. Make it as high as the network is able to handle.
drate = 0

# augmentation probability
# data are shifted in hue, saturation and value with the same probability (but independently)
augmentation_prob = 0.

# mean-variance normalization of incoming samples
# this parameter controls whether mean and variance of images
# should be normalized or not before feeding them to the network
normalize = False


def get_values(hand: dict):
    wrist = np.array([[x[0], x[1]] for x in hand[WRIST]]).flatten()
    thumb = np.array([[x[0], x[1]] for x in hand[THUMB]]).flatten()
    index = np.array([[x[0], x[1]] for x in hand[INDEX]]).flatten()
    middle = np.array([[x[0], x[1]] for x in hand[MIDDLE]]).flatten()
    ring = np.array([[x[0], x[1]] for x in hand[RING]]).flatten()
    pinkie = np.array([[x[0], x[1]] for x in hand[BABY]]).flatten()
    hand_ = np.concatenate((wrist, thumb, index, middle, ring, pinkie)).flatten()
    return hand_


data = DatasetManager(train_samples=10, valid_samples=0, batch_size=10,
                      dataset_dir=joints_path(), formatting=JUNC_STD_FORMAT)

data = data.train()
input_data = data[0][IN(0)]
output_data_heatmaps = data[0][OUT(0)][0]
output_data_visibility = data[0][OUT(1)][0]
print(np.shape(output_data_heatmaps))
print(np.shape(output_data_visibility))
output_data = heatmaps_to_hand(output_data_heatmaps, output_data_visibility)
output_data = get_values(output_data)

print(np.shape(input_data))
print(np.shape(output_data))

if __name__ == '__main__':
    # set_verbosity(COMMENTARY)

    loss = mean_squared_error

    formatting = {
        IN('img'): MIDFMT_JUNC_RGB,
        OUT('heats'): MIDFMT_JUNC_HEATMAP,
        OUT('vis'): MIDFMT_JUNC_VISIBILITY,
    }

    # We need fixed resizing of heatmaps on data read:
    reg_1 = Regularizer().fixresize(52, 52)
    reg_2 = Regularizer().fixresize(200, 200)
    formatting = format_add_outer_func(f=reg_1.apply,
                                       format=formatting,
                                       entry=OUT('heats'))

    formatting = format_add_outer_func(f=reg_2.apply,
                                       format=formatting,
                                       entry=IN('img'))

    formatting = format_add_outer_func(f=lambda x: get_values(heatmaps_to_hand(x, np.zeros(shape=(21,)))),
                                       format=formatting, entry=OUT('heats'))

    # Load data
    dm = DatasetManager(train_samples=train_samples,
                        valid_samples=valid_samples,
                        batch_size=batch_size,
                        dataset_dir=joints_path(),
                        formatting=formatting)

    # Plan the processing needed before providing inputs and outputs for training and validation
    data_processing_plan = ProcessingPlan(augmenter=Augmenter().shift_hue(augmentation_prob)
                                          .shift_sat(augmentation_prob)
                                          .shift_val(augmentation_prob),
                                          regularizer=Regularizer().normalize() if normalize else None,
                                          keyset={IN('img')})
    data_processing_plan.add_outer(key=IN('img'), fun=lambda x: preprocess_mobile(255*x))

    model = train_model(model_generator=lambda: regressor(input_shape=np.shape(dm.train()[0][IN('img')][0])),
                        dataset_manager=dm,
                        loss={OUT('heats'): loss},
                        learning_rate=learning_rate,
                        patience=patience,
                        data_processing_plan=data_processing_plan,
                        tb_path="joints",
                        model_name=model,
                        model_path=joint_locators_path(),
                        epochs=epochs,
                        enable_telegram_log=False)
