import tensorflow as tf
from neural_network.tensorflow_interface.abstract_layer import AbsLayer


class ActivationLayer(AbsLayer):
    def __init__(self, act_func=lambda x, *args, **kwargs: x, args=(), scope=None,
                 tb_manager=None, *extra_args, **extra_kwargs):
        AbsLayer.__init__(self)
        self.__act_func = act_func
        self.__args = args+extra_args
        self.__argdict = extra_kwargs
        self.scope_name = scope
        self.tensor_board_manager = tb_manager

    def make_layer(self, inputs):
        with tf.name_scope(self.scope_name):
            self.set_ready()
            self.output = self.__act_func(inputs, name=self.scope_name, *self.__args, **self.__argdict)
            if self.tensor_board_manager is not None:
                self.tensor_board_manager.add_histogram(self.output, 'output')
