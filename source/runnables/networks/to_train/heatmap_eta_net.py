import os
import sys

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..", "..")))

from library.neural_network.keras.custom_layers.heatmap_loss import *
from data import *
from library import *
import keras as K
from library.neural_network.keras.models.eta_net import eta_net
from library.neural_network.keras.training.model_trainer import train_model
import keras.regularizers as kr
import numpy as np
from library.neural_network.batch_processing.processing_plan import ProcessingPlan

train_samples = 1000
valid_samples = 250
batch_size = 20

if __name__ == '__main__':
    model = 'cropper_eta_net_v1'
    set_verbosity(COMMENTARY)
    m1_path = cropper_h5_path(model)

    # Hyper parameters
    weight_decay = kr.l2(1e-5)
    learning_rate = 1e-4
    white_priority = -2.
    delta = 6
    drate = 0.2

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
                               formatting=formatting)

    # Plan the processing needed before providing inputs and outputs for training and validation
    data_processing_plan = ProcessingPlan(augmenter=Augmenter().shift_hue(.2).shift_sat(.2).shift_val(.2),
                                          keyset={IN(0)})  # Today we just need to augment one input...

    for white_priority in [-2.]:
        model1 = train_model(model_generator=lambda: eta_net(input_shape=np.shape(generator.train()[0][IN(0)])[1:],
                                                             weight_decay=weight_decay,
                                                             dropout_rate=drate,
                                                             activation=lambda: K.layers.LeakyReLU(alpha=0.1)),
                             dataset_manager=generator,
                             loss={OUT(0): lambda x, y: prop_heatmap_penalized_fp_loss(x, y,
                                                                                       white_priority=white_priority,
                                                                                       delta=delta)
                                   },
                             learning_rate=learning_rate,
                             patience=10,
                             data_processing_plan=data_processing_plan,
                             tb_path="heat_maps/",
                             model_name=model + "_nullaugnorm_wp_" + str(white_priority),
                             model_type=CROPPER,
                             epochs=50,
                             enable_telegram_log=False)
