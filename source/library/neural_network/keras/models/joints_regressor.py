import keras.models as km
import keras.layers as kl
from data import *


def regressor(input_shape, weight_decay=None, dropout_rate=0):
    inputs = kl.Input(shape=input_shape, name=IN('img'))
    c1 = kl.Conv2D(filters=32, kernel_size=[5, 5], padding='same', activation='relu')(inputs)
    c2 = kl.Conv2D(filters=64, kernel_size=[5, 5], padding='same', activation='relu',
                   kernel_regularizer=weight_decay)(c1)
    c3 = kl.Conv2D(filters=64, kernel_size=[5, 5], padding='same', activation='relu',
                   kernel_regularizer=weight_decay)(c2)
    p1 = kl.MaxPool2D()(c3)
    c4 = kl.Conv2D(filters=128, kernel_size=[5, 5], padding='same', activation='relu',
                   kernel_regularizer=weight_decay)(p1)
    c5 = kl.Conv2D(filters=128, kernel_size=[5, 5], padding='same', activation='relu',
                   kernel_regularizer=weight_decay)(c4)
    c6 = kl.Conv2D(filters=128, kernel_size=[5, 5], padding='same', activation='relu',
                   kernel_regularizer=weight_decay)(c5)
    p2 = kl.MaxPool2D()(c6)
    c7 = kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation='relu',
                   kernel_regularizer=weight_decay)(p2)
    c8 = kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation='relu',
                   kernel_regularizer=weight_decay)(c7)
    c9 = kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation='relu',
                   kernel_regularizer=weight_decay)(c8)
    p3 = kl.MaxPool2D()(c9)
    f = kl.Flatten()(p3)
    d1 = kl.Dense(units=1024, activation='relu', kernel_regularizer=weight_decay)(f)
    d2 = kl.Dense(units=1024, activation='relu', kernel_regularizer=weight_decay)(d1)
    do = kl.Dropout(rate=dropout_rate)(d2)
    d3 = kl.Dense(units=256, activation='relu', kernel_regularizer=weight_decay)(do)
    out = kl.Dense(units=42, activation='sigmoid', name=OUT('heats'))(d3)

    return km.Model(inputs=[inputs], outputs=[out], name='joints_regressor')
