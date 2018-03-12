import numpy as np
import keras.backend as K
import tensorflow as tf


def heatmap_loss(y_true, y_pred, white_weight=1, black_weight=0.1):
    y_data = tf.reshape(y_true, [-1])
    y_predictions = tf.reshape(y_pred, [-1])
    loss = 0

    return loss


def heatmap_loss_2(y_true, y_pred, white_weight=1, black_weight=0.1):
    return white_weight * K.mean(K.maximum((y_true - y_pred), 0.), axis=-1) \
           + black_weight * K.mean(K.maximum((y_pred - y_true), 0.), axis=-1)
