from library.neural_network.keras.models.building_blocks import serial_pool_reduction_block
from keras import layers as kl
from keras import models as km
from data import *


def normalized_convs(input_shape, activation='relu', name='norm_conv',
                     weight_decay=None, dropout_rate=0):
    inputs = kl.Input(shape=input_shape, name=IN(0))
    first_segment = serial_pool_reduction_block(in_tensor=inputs,
                                                end_ch=256,
                                                topology=(3,),
                                                kernel_size=[5, 5],
                                                activation=activation,
                                                normalize=False,
                                                kreg=weight_decay)
    first_segment = kl.BatchNormalization()(first_segment)
    first_segment = kl.Dropout(rate=dropout_rate)(first_segment)
    second_segment = serial_pool_reduction_block(in_tensor=first_segment,
                                                 end_ch=128,
                                                 topology=(3,),
                                                 kernel_size=[3, 3],
                                                 activation=activation,
                                                 normalize=False,
                                                 kreg=weight_decay)
    second_segment = kl.BatchNormalization()(second_segment)
    second_segment = kl.Dropout(rate=dropout_rate)(second_segment)
    conv_o1 = kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same',
                        kernel_regularizer=weight_decay)(second_segment)
    conv_o1 = kl.Conv2D(filters=64, kernel_size=[3, 3], padding='same',
                        kernel_regularizer=weight_decay)(conv_o1)
    conv_o1 = kl.Conv2D(filters=32, kernel_size=[3, 3], padding='same',
                        kernel_regularizer=weight_decay)(conv_o1)
    out = kl.Conv2D(filters=1, kernel_size=[3, 3], padding='same', name=OUT(0),
                    kernel_regularizer=weight_decay)(conv_o1)

    return km.Model(inputs=[inputs], outputs=[out], name=name)