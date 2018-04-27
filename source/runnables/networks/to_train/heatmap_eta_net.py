import os
import sys

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..", "..")))

from library.neural_network.keras.custom_layers.heatmap_loss import *
from library.neural_network.tensorboard_interface.tensorboard_manager import TensorBoardManager as TBManager
from data.naming import *
import keras as K
from library.neural_network.keras.models.eta_net import eta_net
from library.neural_network.keras.training.model_trainer import train_model
import keras.regularizers as kr
import numpy as np
from data import DatasetManager
from data import Regularizer

train_samples = 500
valid_samples = 200
batch_size = 15

if __name__ == '__main__':
    model = 'cropper_eta_net_v1'
    set_verbosity(COMMENTARY)
    m1_path = cropper_h5_path(model)

    TBManager.set_path("heat_maps")
    train = True

    # Hyper parameters
    weight_decay = kr.l2(1e-5)
    learning_rate = 1e-4

    reg = Regularizer().fixresize(60, 80)
    formatting = CROPS_STD_FORMAT
    oldf = formatting[TARGET][0][1]
    formatting[TARGET] = ((formatting[TARGET][0][0], lambda x: np.expand_dims(reg.apply(oldf(x)), axis=-1)),)
    # Load data
    generator = DatasetManager(train_samples=train_samples,
                               valid_samples=valid_samples,
                               batch_size=batch_size,
                               dataset_dir=crops_path(),
                               formatting=formatting)

    model1 = train_model(model_generator=lambda: eta_net(input_shape=np.shape(generator.train()[0][IN])[1:],
                                                         weight_decay=weight_decay,
                                                         dropout_rate=0.5,
                                                         activation=lambda: K.layers.LeakyReLU(alpha=0.1)),
                         dataset_manager=generator,
                         loss=lambda x, y: prop_heatmap_penalized_fp_loss(x, y, -1.85, 3),
                         learning_rate=learning_rate,
                         patience=5,
                         tb_path="heat_maps/" + model,
                         model_name=model,
                         model_type=CROPPER,
                         epochs=1,
                         verbose=True)
