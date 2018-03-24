import tensorflow as tf


def con_layer(inputs, size_out, window_shape=None, stride_shape=None,
              con_func=tf.nn.conv2d, act_func=tf.nn.relu, name='con', tb_manager=None):

    if stride_shape is None:
        stride_shape = [1, 1, 1, 1]
    if window_shape is None:
        window_shape = [5, 5]

    size_in = int(inputs.shape[-1])
    if tb_manager is not None:
        scope_name = tb_manager.current_scope + name
    else:
        scope_name = name
    with tf.name_scope(scope_name):
        weights = tf.Variable(tf.random_uniform(shape=window_shape+[size_in, size_out], maxval=1, minval=-1), name=con_func.__name__+'weights')
        biases = tf.Variable(tf.random_uniform([size_out], maxval=1), name=con_func.__name__+'biases')
        con = con_func(inputs, weights, strides=stride_shape, padding="SAME")
        act = act_func(con + biases)
        if tb_manager is not None:
            tb_manager.add_histogram(weights, 'weights')
            tb_manager.add_histogram(biases, 'biases')
            if act.shape[-1] < 3:
                images = act[:, :, :, 0:1]
            else:
                images = act[:, :, :, 0:3]
            tb_manager.add_images(images, 'activation_function', img_format=images.shape, max_out=10, collections='c', family='f')
        return act


def polling_layer(inputs, func=tf.nn.max_pool, k_size=None, strider=None, padding='SAME', name='polling'):
    if strider is None:
        strider = [1, 2, 2, 1]
    if k_size is None:
        k_size = [1, 2, 2, 1]

    with tf.name_scope(name):
        pol = func(inputs, ksize=k_size, strides=strider, padding=padding)
    return pol


def drop_out_layer(inputs, keep_probability=0.5, name='dropout'):
    """
    Defines a drop out layer that power off some neurons with a given probability
    :param inputs: are the inputs coming from one layer
    :param keep_probability: is the probability to keep the data coming from a neuron,
                             the drop probability of the the neuron is (1-kp)
    :param name: is the name of the layer
    :return: the result of dropout
    """
    return tf.nn.dropout(inputs, keep_prob=keep_probability, name=name)


def fc_layer(inputs, channel_out, channel_in=None, func=tf.nn.relu, name='fc',
             w_name='W', b_name='B', tb_manager=None):
    """
    Creates a new fully connected layer for the neural network
    :param inputs:  are the effective inputs of the layer
    :param channel_out: is the number of neurons (output) of the layer
    :param channel_in: is the number of inputs in case is not defined in the input variable
    :param func: is the activation function
    :param name: is the name scope of the layer (by default 'fc')
    :param w_name: is the name of weights in the graph (by default 'w')
    :param b_name: is the name ob biases in the graph (by default 'b')
    :param tb_manager: is tensor board manager with all the summaries of the training phase
                                it is initialized to None
    :return: an array with the output of the layer, the weights and the biases
    """
    if tb_manager is not None:
        scope_name = tb_manager.current_scope + name
    else:
        scope_name = name
    with tf.name_scope(scope_name):
        if channel_in is None:
            n_inputs = int(inputs.shape[1])
        else:
            n_inputs = channel_in
        weights = tf.Variable(tf.truncated_normal([n_inputs, channel_out], stddev=0.1), name=w_name)
        bias = tf.Variable(tf.truncated_normal(shape=[channel_out], stddev=0.1), name=b_name)
        output = func(tf.add(tf.matmul(inputs, weights), bias))
        if tb_manager is not None:
            tb_manager.add_histogram(weights, 'weights')
            tb_manager.add_histogram(bias, 'biases')
            tb_manager.add_histogram(output, 'activation_function')
        return output, weights, bias


def loss_function(inputs, exp_out, channel_in=None, channel_out=None,
                  output_func=tf.nn.softmax, func=tf.nn.softmax_cross_entropy_with_logits,
                  loss_const=0.00005, name='output', w_name='W', b_name='B', tb_manager=None):
    """
    Creates a fully connected layer for compute the outputs of the neural network
    and computes the loss function
    :param inputs: are the inputs of the fully connected layer
    :param exp_out: is the desired output
    :param channel_in: is needed if the dimension of the inputs is not specified
                       in the inputs parameter
    :param channel_out: is needed if the dimension of the output is not specified in the
                        exp_out parameter
    :param output_func: is the function for compute the network outputs, by default
                        is used the soft-max
    :param func:  if the loss function, by default, for classification, is used
                  the soft-max cross entropy
    :param loss_const: is the constant of the weights decay
    :param name: is the name scope of the layer (by default 'output')
    :param w_name: is the name of weights in the graph (by default 'w')
    :param b_name: is the name ob biases in the graph (by default 'b')
    :param tb_manager: is a tensor board manager with all the summaries of the training phase
                                it is initialized to None
    :return: an array with the loss function, the output of the layer, the weights and the biases
    """
    if tb_manager is not None:
        scope_name = tb_manager.current_scope + name
    else:
        scope_name = name
    with tf.name_scope(scope_name):
        if channel_in is None:
            n_inputs = int(inputs.shape[1])
        else:
            n_inputs = channel_in

        if channel_out is None:
            n_outputs = int(exp_out.shape[1])
        else:
            n_outputs = channel_out

        weights = tf.Variable(tf.truncated_normal([n_inputs, n_outputs], stddev=0.1), name=w_name)
        bias = tf.Variable(tf.truncated_normal(shape=[+n_outputs], stddev=0.1), name=b_name)
        layer_output = tf.add(tf.matmul(inputs, weights), bias)
        soft_output = output_func(layer_output)
        loss = tf.reduce_mean(func(labels=exp_out, logits=layer_output))
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        computed_loss = loss + loss_const * __get_mean_from_collection(variables)
        gradient = __get_gradient_norm(variables, computed_loss)
        if tb_manager is not None:
            tb_manager.add_histogram(weights, 'weights')
            tb_manager.add_histogram(bias, 'biases')
            tb_manager.add_histogram(soft_output, 'outputs')
            tb_manager.add_scalar(computed_loss, name)
            tb_manager.add_scalar(tf.norm(gradient), 'gradient')
        return computed_loss, soft_output, weights, bias


def fc_neural_network(layers, neurons, inputs, func=tf.nn.tanh, tb_manager=None, name='fc_neural_network'):
    """
    Creates a new neural network with only the hidden layers
    :param layers: is the number of layers
    :param neurons: is the number of neurons per layer
    :param inputs: is the data_set in input
    :param func: is the activation function of the neurons
    :param tb_manager: is the manager for save tensor board statistics
    :param name: is the name space
    :return: the output of the network
    """

    if layers == 0:
        return inputs

    if tb_manager is None:
        name1 = name
    else:
        name1 = tb_manager.current_scope

    layer_name = name1 + func.__name__

    outputs = []
    for l in range(layers):
        if l is 0:
            outputs.append(fc_layer(inputs, neurons, func=func, tb_manager=tb_manager, name=layer_name)[0])
        outputs.append(fc_layer(outputs[-1], neurons, func=func, tb_manager=tb_manager, name=name1)[0])
    return outputs[-1]


def __get_mean_from_collection(collection):
    mean = []
    for e in collection:
        temp = tf.reshape(e, shape=[-1])
        mean.append(tf.norm(temp))
    return tf.reduce_mean(mean)


def __get_gradient_norm(collection, loss):
    temp = 0
    for e in collection:
        temp += tf.norm(tf.gradients(loss, e)) ** 2
    return tf.sqrt(temp)
