from keras.callbacks import Callback
from library.neural_network.tensorboard_interface.tensorboard_manager import TensorBoardManager as TBManager
import tensorflow as tf
import numpy as np
import keras.models as km
from data.augmentation.data_augmenter import Augmenter
import keras.backend as K


class CrossAugmentation(Callback):
    def __init__(self, augmenter: Augmenter=None):
        super(CrossAugmentation, self).__init__()
        # Creating the augmenter
        if augmenter is None:
            self.augmenter = Augmenter().shift_val(0.2).shift_sat(0.2).shift_hue(0.2)
        else:
            self.augmenter = augmenter

    def on_batch_begin(self, batch, logs=None):
        print("Print from callback {}\n\t\t{}".format(np.shape(batch), batch))
        print("Print from callback {}\n\t\t{}".format(np.shape(self.model.inputs), self.model.inputs))
