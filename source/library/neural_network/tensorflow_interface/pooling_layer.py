from neural_network.tensorflow_interface.activation_layer import ActivationLayer
import tensorflow as tf


class PoolingLayer(ActivationLayer):
    def __init__(self, act_func=tf.nn.max_pool, ksize=None, stride=None, scope='Pooling',
                 tb_manager=None):

        if stride is None:
            stride = [1, 2, 2, 1]
        if ksize is None:
            ksize = [1, 2, 2, 1]

        ActivationLayer.__init__(self, act_func, ksize=ksize, strides=stride, padding='SAME',
                      tb_manager=tb_manager, scope=scope)
