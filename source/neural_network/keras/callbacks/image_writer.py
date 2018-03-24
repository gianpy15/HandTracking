from keras.callbacks import Callback
from tensorboard.tensorboard_manager import TensorBoardManager as TBManager
import tensorflow as tf


class ImageWriter(Callback):
    def __init__(self, images=None, tb_manager=None, max_imgs=5):
        super(ImageWriter, self).__init__()
        self.input_images = self.validation_data if images is None else images
        self.tb_manager = TBManager('images') if tb_manager is None else tb_manager
        self.max_imgs = max_imgs
        self.input_images = self.input_images[0:self.max_imgs]

    def on_train_begin(self, logs=None):
        self.tb_manager.add_images(self.model.predict(self.input_images), name='output', max_out=self.max_imgs)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        with tf.Session() as s:
            summary = s.run(self.tb_manager.get_runnable())[0]
            self.tb_manager.write_step(summary, epoch)
