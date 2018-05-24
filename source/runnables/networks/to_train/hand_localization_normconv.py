import os
import sys

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..", "..")))

from library.neural_network.keras.custom_layers.heatmap_loss import *
from data import *
from library import *
import keras as K
from library.neural_network.keras.models.normalized_convolutions import normalized_convs
from library.neural_network.keras.training.model_trainer import train_model
import numpy as np
import keras.regularizers as kr
from library.neural_network.batch_processing.processing_plan import ProcessingPlan
from library.utils.visualization_utils import get_image_with_mask, crop_sprite
from data.datasets.crop.crop_exclude import multiple_hands_video_list


# ####################### HYPERPARAMETERS #######################

# NETWORK NAME (used for naming saved files)
# play with this adding extentions when saving models with different hyperparameter configurations
model = 'normalized_leaky_convs'

# TRAINING PARAMETERS:

# the number of training samples loaded
train_samples = 1  # >=1

# the number of validation samples loaded
valid_samples = 1  # >=1

# the number of samples used for each batch
# higher batch size leads to more significant gradient (less variance in gradient)
# but a batch size too high may not fit into the graphics video memory.
batch_size = 20  # >=1

# the number of epochs to perform without improvements in validation accuracy before triggering early stopping
# higher patience allows bridging greater "hills" but with obvious downsides in case the overfitting hill never ends
patience = 10  # >=1

# the maximum number of epochs to perform before stopping.
# notice: usually this term is not influential because early stopping triggers first
epochs = 1  # >=1

# learning rate used for optimization
# higher learning rates lead to definitely faster convergence but possibly unstable behaviour
# setting a lower learning rate may also allow reaching smaller and deeper minima in the loss
# use it to save training time, but don't abuse it as you may lose the best solutions
learning_rate = 1e-4  # >0

# LOSS PARAMETERS

# the extra importance to give to whites in target heatmaps.
# This heatmap loss function is internally auto-balanced to make whites and blacks equally important
# in the target heatmaps even when they appear in different proportions.
# This parameter changes a little bit the equilibrium favouring the white (when positive)
# This may solve the problem of the network outputting either full-black or full-white heatmaps
white_priority = -2.  # any value, 0 is perfect equilibrium

# the extra importance to give to false positives.
# Use this parameter to increase the penalty of saying there is a hand where there is not.
# The penalty is proportional and additive: delta=6 means we will add 6 times the penalty for false positives.
delta = 6  # >=-1, 0 is not additional penalty, -1<delta<0 values discount penalty. delta<=-1 PROMOTES MISTAKES.


# NETWORK PARAMETERS

# the dropout rate to be used in the entire network
# dropout will make training and learning more difficult as it shuts down random units at training time
# but it will improve generalization a lot. Make it as high as the network is able to handle.
drate = 0.2

# leaky relu coefficient
# relu is great, but sometimes it leads to "neuron death": a neuron jumps into the flat zero region, then
# it will always have zero gradient and will never be able to recover back if needed.
# For this reason leaky relu exists, and this parameter encodes the slope of the negative part of the activation.
leaky_slope = 0.1  # >=0, 0 is equivalent to relu, 1 is equivalent to linear, higher is possible but not recommended

# augmentation probability
# data are shifted in hue, saturation and value with the same probability (but independently)
augmentation_prob = 0.2

# mean-variance normalization of incoming samples
# this parameter controls whether mean and variance of images
# should be normalized or not before feeding them to the network
normalize = False

# weight decay
# the regularization technique used for the network.
# kr.l2 is the L2 norm (sum of squares) and favours accuracy on models whose weights should be gaussian
# kr.l1 is the L1 norm (sum of absolute values) and favours sparse models
weight_decay = kr.l2(1e-5)

if __name__ == '__main__':
    set_verbosity(DEBUG)
    m1_path = cropper_h5_path(model)

    # We need fixed resizing of heatmaps on data read:
    reg = Regularizer().fixresize(60, 80)
    formatting = format_add_outer_func(f=reg.apply,
                                       format=CROPS_STD_FORMAT,
                                       entry=OUT(0))
    # Load data
    generator = DatasetManager(train_samples=train_samples,
                               valid_samples=valid_samples,
                               batch_size=batch_size,
                               dataset_dir=crops_path(),
                               formatting=formatting,
                               exclude_videos=multiple_hands_video_list())

    # Plan the processing needed before providing inputs and outputs for training and validation
    data_processing_plan = ProcessingPlan(augmenter=Augmenter().shift_hue(augmentation_prob)
                                          .shift_sat(augmentation_prob)
                                          .shift_val(augmentation_prob),
                                          regularizer=Regularizer().normalize() if normalize else None,
                                          keyset={IN(0)})  # Today we just need to augment one input...
    model1 = train_model(model_generator=lambda: normalized_convs(input_shape=np.shape(generator.train()[0][IN(0)])[1:],
                                                                  dropout_rate=drate,
                                                                  weight_decay=weight_decay,
                                                                  activation=lambda: K.layers.LeakyReLU(alpha=leaky_slope)),
                         dataset_manager=generator,
                         loss={OUT(0): lambda x, y: prop_heatmap_penalized_fp_loss(x, y,
                                                                                   white_priority=white_priority,
                                                                                   delta=delta)
                               },
                         learning_rate=learning_rate,
                         patience=patience,
                         data_processing_plan=data_processing_plan,
                         tb_path="heat_maps/",
                         tb_plots={'plain_input': lambda feed: feed[IN(0)],
                                   'plain_target': lambda feed: feed[OUT(0)],
                                   'plain_output': lambda feed: feed[NET_OUT(0)],
                                   'combined_mask': lambda feed: get_image_with_mask(feed[IN(0)],
                                                                                     feed[NET_OUT(0)]),
                                   'crops': crop_sprite},
                         model_name=model,
                         model_path=croppers_path(),
                         epochs=epochs,
                         enable_telegram_log=True)
