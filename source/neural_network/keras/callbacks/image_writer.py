from keras.callbacks import Callback
from tensorboard_utils.tensorboard_manager import TensorBoardManager as TBManager
import tensorflow as tf
import numpy as np


class ImageWriter(Callback):
    def __init__(self, images=None, tb_manager=None, max_imgs=5, name='images'):
        super(ImageWriter, self).__init__()
        self.input_images = self.validation_data if images is None else images
        self.tb_manager = TBManager(name) if tb_manager is None else tb_manager
        self.name = name
        self.max_imgs = max_imgs
        self.input_images = self.input_images[0:self.max_imgs]
        self.image_tensor = None

    def on_train_begin(self, logs=None):
        self.image_tensor = tf.placeholder(dtype=tf.float32, shape=np.shape(self.model.predict(self.input_images)))
        self.tb_manager.add_images(self.image_tensor, name=self.name, max_out=self.max_imgs)
        # self.tb_manager.add_images(self.model.predict(self.input_images), name='output', max_out=self.max_imgs)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        heat_maps = self.model.predict(self.input_images)
        with tf.Session() as s:
            summary = s.run(self.tb_manager.get_runnable(),
                            feed_dict={self.image_tensor: heat_maps})[0]
            self.tb_manager.write_step(summary, epoch)
