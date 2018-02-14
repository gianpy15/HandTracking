from neural_network.abstract_layer import AbsLayer
import tensorflow as tf


class LossExtractorLayer(AbsLayer):
    def __init__(self, expected_output, output_func=tf.nn.softmax, loss_func=tf.nn.softmax_cross_entropy_with_logits,
                 weight_decay=0, decay_coll=tf.GraphKeys.TRAINABLE_VARIABLES, norm_fun=None, scope='Loss', tb_manager=None):
        AbsLayer.__init__(self)
        self.scope_name = scope
        self.tensor_board_manager = tb_manager
        self.loss_fun = loss_func
        self.output_fun = output_func
        self.weight_decay_coeff = weight_decay
        self.expected_output = expected_output
        self.loss = None
        self.decay_coll = decay_coll
        if norm_fun is None:
            self.norm_fun = self.l2_norm
        else:
            self.norm_fun = norm_fun

    @staticmethod
    def l2_norm(collection):
        tmp = []
        for e in collection:
            temp = tf.reshape(e, shape=[-1])
            tmp.append(tf.norm(temp) ** 2)
        return tf.reduce_sum(tmp)

    def make_layer(self, inputs):
        self.set_ready()
        with tf.name_scope(self.scope_name):
            self.output = self.output_fun(inputs, name='Softmax')
            self.loss = tf.reduce_mean(self.loss_fun(labels=self.expected_output, logits=inputs, name='Loss'))
            if self.weight_decay_coeff != 0:
                variables = tf.get_collection(self.decay_coll)
                self.loss += self.weight_decay_coeff * self.norm_fun(variables)
            if self.tensor_board_manager is not None:
                self.tensor_board_manager.add_histogram(self.output, 'net_output')
                self.tensor_board_manager.add_scalar(self.loss, 'net_loss')
