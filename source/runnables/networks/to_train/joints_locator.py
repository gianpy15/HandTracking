import os
import sys

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..", "..")))

from library.neural_network.keras.custom_layers.heatmap_loss import *
from data import *
from library import *
import keras as K
from library.neural_network.keras.models.joints import low_injection_locator
from library.neural_network.keras.training.model_trainer import train_model
from library.neural_network.batch_processing.processing_plan import ProcessingPlan
from library.utils.visualization_utils import joint_skeleton_impression

# ####################### HYPERPARAMETERS #######################

# NETWORK NAME (used for naming saved files)
# play with this adding extentions when saving models with different hyperparameter configurations
model = 'jlocator_lowinj'

# TRAINING PARAMETERS:

# the number of training samples loaded
train_samples = 100  # >=1

# the number of validation samples loaded
valid_samples = 100  # >=1

# the number of samples used for each batch
# higher batch size leads to more significant gradient (less variance in gradient)
# but a batch size too high may not fit into the graphics video memory.
batch_size = 14  # >=1

# the number of epochs to perform without improvements in validation accuracy before triggering early stopping
# higher patience allows bridging greater "hills" but with obvious downsides in case the overfitting hill never ends
patience = 10  # >=1

# the maximum number of epochs to perform before stopping.
# notice: usually this term is not influential because early stopping triggers first
epochs = 20  # >=1

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


# #################### TRAINING #########################

if __name__ == '__main__':
    set_verbosity(COMMENTARY)

    heatmap_loss = lambda x, y: prop_heatmap_penalized_fp_loss(x, y,
                                                               white_priority=white_priority,
                                                               delta=delta)

    JUNC_LOWINJ_FORMAT = {
        IN('img'): MIDFMT_JUNC_RGB,
        OUT('mid_heats'): MIDFMT_JUNC_HEATMAP,
        OUT('vis'): MIDFMT_JUNC_VISIBILITY,
        OUT('heats'): MIDFMT_JUNC_HEATMAP
    }

    def reduce_heatmap_by_two(heat: np.ndarray):
        heatshape = np.shape(heat)
        outheat = np.zeros(shape=(int(heatshape[0] / 2), int(heatshape[1] / 2), heatshape[2]), dtype=heat.dtype)
        for row in range(len(heat)):
            for col in range(len(heat[row])):
                outheat[int(row/2), int(col/2)] += heat[row, col] / 4
        return outheat

    JUNC_LOWINJ_FORMAT = format_add_outer_func(f=reduce_heatmap_by_two,
                                               format=JUNC_LOWINJ_FORMAT,
                                               entry=OUT('heats'))

    # Load data
    dm = DatasetManager(train_samples=train_samples,
                        valid_samples=valid_samples,
                        batch_size=batch_size,
                        dataset_dir=joints_path(),
                        formatting=JUNC_LOWINJ_FORMAT)

    # Plan the processing needed before providing inputs and outputs for training and validation
    data_processing_plan = ProcessingPlan(augmenter=Augmenter().shift_hue(augmentation_prob)
                                          .shift_sat(augmentation_prob)
                                          .shift_val(augmentation_prob),
                                          regularizer=Regularizer().normalize() if normalize else None,
                                          keyset={IN('img')})  # Today we just need to augment one input...
    model = train_model(model_generator=lambda: low_injection_locator(input_shape=np.shape(dm.train()[0][IN('img')][0]),
                                                                      dropout_rate=drate,
                                                                      activation=lambda: K.layers.LeakyReLU(alpha=leaky_slope)
                                                                      ),
                        dataset_manager=dm,
                        loss={OUT('heats'): heatmap_loss,
                              OUT('mid_heats'): heatmap_loss,
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
                                  'mid_output': lambda feed: joint_skeleton_impression(feed,
                                                                                       img_key=IN('img'),
                                                                                       heats_key=NET_OUT('mid_heats'),
                                                                                       vis_key=NET_OUT('vis'))
                                  },
                        model_name=model,
                        model_path=joint_locators_path(),
                        epochs=epochs,
                        enable_telegram_log=False)
