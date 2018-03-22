import keras.models as km
import keras.layers as kl
import keras.callbacks as kc
import keras.optimizers as ko
import keras.regularizers as kr
import matplotlib.pyplot as plt
import numpy as np


def simple_model(input_shape, weight_decay=0):
    model = km.Sequential()
    model.add(kl.Conv2D(input_shape=input_shape, filters=32, kernel_size=[3, 3], padding='same'))
    model.add(kl.Activation('relu'))
    model.add(kl.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.MaxPooling2D())
    model.add(kl.Conv2D(filters=16, kernel_size=[3, 3], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.MaxPooling2D())
    model.add(kl.Conv2D(filters=1, kernel_size=[3, 3], padding='same', kernel_regularizer=weight_decay))
    # model.add(Softmax4D(axis=1, name='softmax4D'))
    model.add(kl.Activation('sigmoid'))

    return model


def exotic_model(input_shape, weight_decay=0):
    model = km.Sequential()
    model.add(kl.Conv2D(input_shape=input_shape, filters=32, kernel_size=[3, 3], padding='same'))
    model.add(kl.Activation('relu'))
    # encoding
    model.add(kl.Conv2D(filters=64, kernel_size=[3, 3], strides=[2, 2], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.Conv2D(filters=128, kernel_size=[3, 3], strides=[2, 2], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.Conv2D(filters=256, kernel_size=[3, 3], strides=[2, 2], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    # decoding
    model.add(kl.Conv2DTranspose(filters=256, kernel_size=[3, 3], strides=[2, 2], padding='same', activation='relu',
                                 kernel_regularizer=weight_decay))
    model.add(kl.Conv2DTranspose(filters=128, kernel_size=[3, 3], strides=[2, 2], padding='same', activation='relu',
                                 kernel_regularizer=weight_decay))
    model.add(kl.Conv2DTranspose(filters=64, kernel_size=[3, 3], strides=[2, 2], padding='same', activation='relu',
                                 kernel_regularizer=weight_decay))

    model.add(kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.MaxPooling2D())
    model.add(kl.Conv2D(filters=16, kernel_size=[3, 3], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.MaxPooling2D())
    model.add(kl.Conv2D(filters=1, kernel_size=[3, 3], padding='same', kernel_regularizer=weight_decay))
    # model.add(Softmax4D(axis=1, name='softmax4D'))
    model.add(kl.Activation('sigmoid'))

    return model
