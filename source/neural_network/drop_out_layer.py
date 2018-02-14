from neural_network.activation_layer import ActivationLayer
import tensorflow as tf


class DropOut(ActivationLayer):
    def __init__(self, act_func=tf.nn.dropout, keep_prob=0.5, noise_shape=None, scope='Dropout', tb_manager=None):

        ActivationLayer.__init__(self, act_func=act_func, keep_prob=keep_prob, noise_shape=noise_shape, scope=scope, tb_manager=tb_manager)
