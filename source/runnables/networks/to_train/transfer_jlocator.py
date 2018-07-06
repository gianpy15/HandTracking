import os
import sys

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..", "..")))

from library.neural_network.keras.custom_layers.heatmap_loss import *
from data import *
from library import *
import keras as K
from library.neural_network.keras.models.transfer_learning import transfer_mobile_net_joints
from library.neural_network.keras.training.model_trainer import train_model
from library.neural_network.batch_processing.processing_plan import ProcessingPlan
from library.utils.visualization_utils import joint_skeleton_impression
from keras.applications.mobilenet import preprocess_input as preprocess_mobile

# ####################### HYPERPARAMETERS #######################

# NETWORK NAME (used for naming saved files)
# play with this adding extentions when saving models with different hyperparameter configurations
model = 'transfer_jlocator'

# TRAINING PARAMETERS:

# the number of training samples loaded
train_samples = 500  # >=1

# the number of validation samples loaded
valid_samples = 10  # >=1

# the number of samples used for each batch
# higher batch size leads to more significant gradient (less variance in gradient)
# but a batch size too high may not fit into the graphics video memory.
batch_size = 15  # >=1

# the number of epochs to perform without improvements in validation accuracy before triggering early stopping
# higher patience allows bridging greater "hills" but with obvious downsides in case the overfitting hill never ends
patience = 1000  # >=1

# the maximum number of epochs to perform before stopping.
# notice: usually this term is not influential because early stopping triggers first
epochs = 500  # >=1

# learning rate used for optimization
# higher learning rates lead to definitely faster convergence but possibly unstable behaviour
# setting a lower learning rate may also allow reaching smaller and deeper minima in the loss
# use it to save training time, but don't abuse it as you may lose the best solutions
learning_rate = 1e-5  # >0

# LOSS PARAMETERS

# the extra importance to give to whites in target heatmaps.
# This heatmap loss function is internally auto-balanced to make whites and blacks equally important
# in the target heatmaps even when they appear in different proportions.
# This parameter changes a little bit the equilibrium favouring the white (when positive)
# This may solve the problem of the network outputting either full-black or full-white heatmaps
white_priority = -.2  # any value, 0 is perfect equilibrium

# how much the heatmap loss is scaled up against the visibility loss.
# makes sure that the algorithm gives the right priority to losses
heat_priority = 100

# the extra importance to give to false positives.
# Use this parameter to increase the penalty of saying there is a hand where there is not.
# The penalty is proportional and additive: delta=6 means we will add 6 times the penalty for false positives.
delta = 1  # >=-1, 0 is not additional penalty, -1<delta<0 values discount penalty. delta<=-1 PROMOTES MISTAKES.


# NETWORK PARAMETERS

# the dropout rate to be used in the entire network
# dropout will make training and learning more difficult as it shuts down random units at training time
# but it will improve generalization a lot. Make it as high as the network is able to handle.
drate = 0.1

# leaky relu coefficient
# relu is great, but sometimes it leads to "neuron death": a neuron jumps into the flat zero region, then
# it will always have zero gradient and will never be able to recover back if needed.
# For this reason leaky relu exists, and this parameter encodes the slope of the negative part of the activation.
leaky_slope = 0.1  # >=0, 0 is equivalent to relu, 1 is equivalent to linear, higher is possible but not recommended

# augmentation probability
# data are shifted in hue, saturation and value with the same probability (but independently)
augmentation_prob = 0.

# mean-variance normalization of incoming samples
# this parameter controls whether mean and variance of images
# should be normalized or not before feeding them to the network
normalize = False

# decide whether to retrain transfered mobilenet weights (heavier, more accurate) or not (faster training)
retrain = True


# #################### TRAINING #########################

if __name__ == '__main__':
    # set_verbosity(COMMENTARY)

    heatmap_loss = lambda x, y: heat_priority*prop_heatmap_penalized_fp_loss(x, y,
                                                                             white_priority=white_priority,
                                                                             delta=delta)

    formatting = {
        IN('img'): MIDFMT_JUNC_RGB,
        OUT('heats'): MIDFMT_JUNC_HEATMAP,
        OUT('vis'): MIDFMT_JUNC_VISIBILITY,
    }

    # We need fixed resizing of heatmaps on data read:
    reg_1 = Regularizer().fixresize(56, 56)
    reg_2 = Regularizer().fixresize(224, 224)
    formatting = format_add_outer_func(f=reg_1.apply,
                                       format=formatting,
                                       entry=OUT('heats'))

    formatting = format_add_outer_func(f=reg_2.apply,
                                       format=formatting,
                                       entry=IN('img'))

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

    model = train_model(model_generator=lambda: transfer_mobile_net_joints(# input_shape=np.shape(dm.train()[0][IN('img')][0]),
                                                                          dropout_rate=drate,
                                                                          train_mobilenet=retrain,
                                                                          # activation=K.layers.LeakyReLU(alpha=leaky_slope)
                                                                          ),
                        dataset_manager=dm,
                        loss={OUT('heats'): heatmap_loss,
                              OUT('vis'): 'binary_crossentropy'},
                        learning_rate=learning_rate,
                        patience=patience,
                        data_processing_plan=data_processing_plan,
                        tb_path="joints",
                        tb_plots={'target': lambda feed: joint_skeleton_impression(feed,
                                                                                   img_key=IN('img'),
                                                                                   heats_key=OUT('heats'),
                                                                                   vis_key=OUT('vis')),
                                  'output': lambda feed: joint_skeleton_impression(feed,
                                                                                   img_key=IN('img'),
                                                                                   heats_key=NET_OUT('heats'),
                                                                                   vis_key=NET_OUT('vis')),
                                  },
                        model_name=model,
                        model_path=joint_locators_path(),
                        epochs=epochs,
                        enable_telegram_log=True)