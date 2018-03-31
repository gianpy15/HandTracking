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


def prop_heatmap_loss(heat_ground, heat_pred, white_priority=0):
    """
    Loss function for heatmaps to give priority weight to pixels that are
    far from the mean of the ground truth. This mean can be altered in order
    to further condition the priority.
    :param heat_pred: predicted heatmap, values in [0, 1]
    :param heat_ground: ground truth heatmap, values in [0, 1]
    :param white_priority: further priority to give to white, 0 gives no priority,
        values in [-inf, +inf]
    :return: the normalized weighted loss for the given heatmaps
    """
    # Be X the random variable of one target pixel defined over the pixels of heat_ground
    # E[X]:
    base_mean = K.mean(heat_ground)
    # shift the mean to condition the weights
    # notice that:
    # 0-> black
    # 1-> white
    # sigmoid(-log(1/t -1)) = t
    shifted_mean = K.sigmoid(-K.log(1 / base_mean - 1) - white_priority)

    # Now we want to normalize the resulting weight map:
    # E[(X-E[X] + delta)^2] =
    # = E[(X-E[X])^2 +2*delta*(X-E[X]) + delta^2] =
    # = Var(X) + delta^2
    mean_delta_sqr = K.square(base_mean-shifted_mean)
    norm_factor = K.var(heat_ground) + mean_delta_sqr

    weight_map = K.square(heat_ground-shifted_mean)/norm_factor

    # now apply the weights with component-by-component product
    weighted_loss = K.square(heat_pred-heat_ground)*weight_map

    return weighted_loss


def static_prop_heatmap_parameters(heat_grounds, white_priority):
    """
    Extract the shifted mean and needed normalization statically
    from a given set of sample targets.
    Equivalent to the procedure in prop_heatmap_loss but static.
    :param heat_grounds: set of ground truth targets, values in [0, 1]
    :param white_priority: further priority to give to white, 0 gives no priority,
        values in [-inf, +inf]
    :return: (shifted mean, normalization factor)
    """
    # Be X the random variable of one target pixel defined over the pixels of heat_ground
    # E[X]:
    base_mean = np.mean(heat_grounds)
    # shift the mean to condition the weights
    shifted_mean = 1/(1 + np.exp(np.log(1 / base_mean - 1) + white_priority))

    # Now we want to normalize the resulting weight map:
    # E[(X-E[X] + delta)^2] =
    # = E[(X-E[X])^2 +2*delta*(X-E[X]) + delta^2] =
    # = Var(X) + delta^2
    mean_delta_sqr = np.square(base_mean - shifted_mean)
    norm_factor = np.var(heat_grounds) + mean_delta_sqr

    return shifted_mean, norm_factor


def prop_heatmap_loss_fast(heat_ground, heat_pred, mean_par, norm_factor):
    """
    A fast version of the prop_heatmap_loss that assumes mean and normalization
    precomputed over some samples. See prop_heatmap_loss doc for more info.
    """
    weight_map = K.square(heat_ground - mean_par) / norm_factor

    # now apply the weights with component-by-component product
    weighted_loss = K.square(heat_pred - heat_ground) * weight_map

    return weighted_loss



