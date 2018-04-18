from keras.callbacks import Callback
from library.neural_network.tensorboard_interface.tensorboard_manager import TensorBoardManager as TBManager
import tensorflow as tf
import numpy as np
from library.utils.visualization_utils import get_image_with_mask


class ImageWriter(Callback):
    def __init__(self, data: tuple=(None, None), max_imgs=5, name='images'):
        tb_manager = TBManager(scope_name=name)
        super(ImageWriter, self).__init__()
        self.input_images = data[0][0:max_imgs]
        self.input_images_3d = None
        self.target_images = data[1][0:max_imgs]
        self.tb_manager = TBManager(name) if tb_manager is None else tb_manager
        self.name = name
        self.max_imgs = max_imgs
        self.image_tensor = None
        self.output_tensor = None
        self.target_tensor = None
        self.mask_tensor = None

    def on_train_begin(self, logs=None):
        if self.input_images is not None and self.target_images is not None:
            self.input_images_3d = self.input_images[:, :, :, 0:3]

            self.image_tensor = tf.placeholder(dtype=tf.float32,
                                               shape=np.shape(self.input_images_3d),
                                               name=self.name + "_X")
            self.output_tensor = tf.placeholder(dtype=tf.float32,
                                                shape=np.shape(self.model.predict(self.input_images)),
                                                name=self.name + "_Y")
            self.target_tensor = tf.placeholder(dtype=tf.float32,
                                                shape=np.shape(self.target_images),
                                                name=self.name + "_T")
            self.mask_tensor = tf.placeholder(dtype=tf.float32,
                                              shape=np.shape(self.input_images_3d),
                                              name=self.name + "_Mask")
            self.tb_manager.add_images(self.image_tensor, name=self.name + "_X", max_out=self.max_imgs)
            self.tb_manager.add_images(self.output_tensor, name=self.name + "_Y", max_out=self.max_imgs)
            self.tb_manager.add_images(self.target_tensor, name=self.name + "_T", max_out=self.max_imgs)
            self.tb_manager.add_images(self.mask_tensor, name=self.name + "_Mask", max_out=self.max_imgs)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        if self.input_images is not None and self.target_images is not None:
            heat_maps = self.model.predict(self.input_images)
            with tf.Session() as s:
                summary = s.run(self.tb_manager.get_runnable(),
                                feed_dict={self.image_tensor: self.input_images_3d,
                                           self.output_tensor: heat_maps,
                                           self.target_tensor: self.target_images,
                                           self.mask_tensor: get_image_with_mask(self.input_images_3d, heat_maps)})[0]
                self.tb_manager.write_step(summary, epoch)
