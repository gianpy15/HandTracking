from neural_network.tensorflow_interface.abstract_layer import AbsLayer
import tensorflow as tf
import numpy as np


class ConvolutionLayer(AbsLayer):
    def __init__(self, depth_out=3, stride_shape=None, window_shape=None,
                 act_func=tf.nn.relu, scope=None, padding='SAME',
                 tb_manager=None, var_collection=None, plot_filters=False,
                 conv_func=tf.nn.conv2d, **additional_params):
        AbsLayer.__init__(self)
        self.__act_func = act_func
        self.scope_name = scope
        self.tensor_board_manager = tb_manager
        self.padding = padding
        self.plot_filters = plot_filters
        self.conv_func = conv_func
        self.add_params = additional_params
        if self.conv_func is tf.nn.conv2d:
            if stride_shape is None:
                self.add_params['strides'] = [1, 1, 1, 1]
            else:
                self.add_params['strides'] = stride_shape
        if window_shape is None:
            self.__window_shape = [3, 3]
        else:
            self.__window_shape = window_shape
        self.depth_out = depth_out
        self.var_collection = var_collection
        if self.__act_func == tf.nn.relu:
            self.stddev_coeff = 2
        else:
            self.stddev_coeff = 1

    def make_layer(self, inputs):
        self.set_ready()
        with tf.name_scope(self.scope_name):
            channel_in = int(inputs.shape[-1])
            fan_in = self.__window_shape[0]*self.__window_shape[1]*channel_in
            weights = tf.Variable(tf.random_normal(shape=self.__window_shape + [channel_in, self.depth_out],
                                  name='2Dconv_weights', stddev=np.sqrt(self.stddev_coeff/fan_in)))
            biases = tf.Variable(tf.truncated_normal([self.depth_out], stddev=0.1), name='2Dconv_biases')
            con = self.conv_func(inputs, weights, padding=self.padding, **self.add_params)
            act = self.__act_func(con + biases)
            if self.tensor_board_manager is not None:
                self.tensor_board_manager.add_histogram(weights, 'weights')
                self.tensor_board_manager.add_histogram(biases, 'biases')
                if self.plot_filters:
                    i = 0
                    while i + 3 < self.depth_out:
                        images = act[:, :, :, i:i+3]
                        self.tensor_board_manager.add_images(images, 'activation_function', img_format=images.shape,
                                                             max_out=1,
                                                             collections='COLLECTIONS IS HERE',
                                                             family=self.scope_name)
                        i += 3
            if self.var_collection is not None:
                tf.add_to_collection(self.var_collection, weights)
                tf.add_to_collection(self.var_collection, biases)
        self.output = act
