from keras.callbacks import Callback
from library.neural_network.tensorboard_interface.tensorboard_manager import TensorBoardManager as TBManager
import tensorflow as tf
import numpy as np
import threading


class ScalarWriter(Callback):
    def __init__(self, name='scalars', freq=1):
        self.tb_manager = TBManager(scope_name=name)
        super(ScalarWriter, self).__init__()
        self.name = name
        self.freq = freq
        self.train_loss_tensor = None
        self.valid_loss_tensor = None
        self.train_acc_tensor = None
        self.valid_acc_tensor = None

    def on_train_begin(self, logs=None):
        self.train_loss_tensor = tf.placeholder(dtype=tf.float32,
                                                shape=(),
                                                name="train_loss")
        self.valid_loss_tensor = tf.placeholder(dtype=tf.float32,
                                                shape=(),
                                                name="valid_loss")
        self.train_acc_tensor = tf.placeholder(dtype=tf.float32,
                                               shape=(),
                                               name="train_acc")
        self.valid_acc_tensor = tf.placeholder(dtype=tf.float32,
                                               shape=(),
                                               name="valid_acc")
        self.tb_manager.add_scalar(self.train_loss_tensor, name="train_loss")
        self.tb_manager.add_scalar(self.train_acc_tensor, name="train_accuracy")
        self.tb_manager.add_scalar(self.valid_loss_tensor, name="validation_loss")
        self.tb_manager.add_scalar(self.valid_acc_tensor, name="validation_accuracy")
        self.tb_manager.write_graph(tf.get_default_graph())

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if epoch % self.freq == 0:
            write_thread = threading.Thread(target=self.__write_step,
                                            args=(logs, epoch, tf.get_default_graph()),
                                            daemon=True)
            write_thread.start()

    def __write_step(self, logs, epoch, cur_graph):
        train_loss = logs['loss']
        valid_loss = logs['val_loss']
        train_acc = logs['acc']
        valid_acc = logs['val_acc']

        with tf.Session(graph=cur_graph) as s:
            summary = s.run(self.tb_manager.get_runnable(),
                            feed_dict={self.train_loss_tensor: train_loss,
                                       self.train_acc_tensor: train_acc,
                                       self.valid_loss_tensor: valid_loss,
                                       self.valid_acc_tensor: valid_acc})[0]
            self.tb_manager.write_step(summary, epoch)
