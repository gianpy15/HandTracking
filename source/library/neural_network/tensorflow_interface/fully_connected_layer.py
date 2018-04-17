import tensorflow as tf
from library.neural_network.tensorflow_interface.abstract_layer import AbsLayer
import numpy as np


class FullyConnectedLayer(AbsLayer):
    def __init__(self, shape_out, act_func=tf.nn.relu, scope=None,
                 tb_manager=None, var_collection=None):
        AbsLayer.__init__(self)
        self.shape_out = shape_out
        self.__act_func = act_func
        self.__channel_out = 1
        for dim in self.shape_out:
            self.__channel_out *= dim
        self.scope_name = scope
        self.tensor_board_manager = tb_manager
        self.weights = None
        self.bias = None
        self.var_collection=var_collection
        if self.__act_func == tf.nn.relu:
            self.stddev_coeff = 2
        else:
            self.stddev_coeff = 1

    def make_layer(self, inputs):
        with tf.name_scope(self.scope_name):
            self.set_ready()
            input_size = 1
            for dim in inputs.shape[1:]:
                input_size *= int(dim)
            net_in = tf.reshape(inputs, [-1, input_size])
            n_inputs = int(net_in.shape[1])
            weights = tf.Variable(tf.truncated_normal([n_inputs, self.__channel_out], stddev=np.sqrt(self.stddev_coeff/n_inputs)), name='weights')
            bias = tf.Variable(tf.ones(shape=[self.__channel_out]), name='bias')
            output = self.__act_func(tf.add(tf.matmul(net_in, weights), bias), name=self.scope_name)
            self.output = tf.reshape(output, [-1] + self.shape_out)
            self.weights = weights
            self.bias = bias
            if self.tensor_board_manager is not None:
                self.tensor_board_manager.add_histogram(self.weights, 'weights')
                self.tensor_board_manager.add_histogram(self.bias, 'bias')
                self.tensor_board_manager.add_histogram(self.output, 'output')
            if self.var_collection is not None:
                tf.add_to_collection(self.var_collection, weights)
                tf.add_to_collection(self.var_collection, bias)
