from keras.callbacks import Callback
from tensorboard_utils.tensorboard_manager import TensorBoardManager as TBManager
import tensorflow as tf
import numpy as np


class InputWriter(Callback):
    def __init__(self, tb_manager=None, max_imgs=5, name='input'):
        super(InputWriter, self).__init__()
        self.tb_manager = TBManager(name) if tb_manager is None else tb_manager
        self.name = name
        self.max_imgs = max_imgs
        self.image_tensor = None

    def on_train_begin(self, logs=None):
        imgs = self.model.inputs[0:self.max_imgs, :, :, -1]
        print("******model with input shape: {}".format(np.shape(imgs)))
        self.image_tensor = tf.placeholder(dtype=tf.float32, shape=np.shape(imgs))
        self.tb_manager.add_images(self.image_tensor, name=self.name, max_out=self.max_imgs)
        # self.tb_manager.add_images(self.model.predict(self.input_images), name='output', max_out=self.max_imgs)
