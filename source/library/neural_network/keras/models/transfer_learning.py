import os
import sys

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..", "..", "..")))

import keras as K
import keras.models as km
from data import *
from keras.applications.mobilenet import MobileNet


def transfer_vgg(weight_decay=None, dropout_rate=0.0, activation=K.layers.activations.relu, train_vgg=False):
    vgg = K.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=None)
    layers = vgg.layers[:10]
    for layer in layers:
        layer.trainable = train_vgg
    layers[0].name = IN(0)
    transferred_net = K.models.Sequential(layers=layers)
    transferred_net.add(K.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same',
                                        kernel_regularizer=weight_decay, activation=activation))
    transferred_net.add(K.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same',
                                        kernel_regularizer=weight_decay, activation=activation))
    transferred_net.add(K.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same',
                                        kernel_regularizer=weight_decay, activation=activation))

    transferred_net.add(K.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same',
                                        kernel_regularizer=weight_decay, activation=activation))
    transferred_net.add(K.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same',
                                        kernel_regularizer=weight_decay, activation=activation))
    transferred_net.add(K.layers.SpatialDropout2D(rate=dropout_rate))

    transferred_net.add(K.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='same',
                                        kernel_regularizer=weight_decay, activation=activation))
    transferred_net.add(K.layers.Conv2D(filters=1, kernel_size=[3, 3], padding='same',
                                        kernel_regularizer=weight_decay, activation='sigmoid'))

    transferred_net.layers[-1].name = OUT(0)

    return transferred_net


def transfer_mobile_net(dropout_rate=0.0, activation=K.layers.activations.relu, train_mobilenet=False):
    mobile_net = MobileNet(include_top=False, weights='imagenet', dropout=0.0, input_shape=[224, 224, 3])
    layers = mobile_net.layers[:26]
    for layer in layers:
        layer.trainable = train_mobilenet
    layers[0].name = IN(0)

    transferred_net = K.models.Sequential(layers=layers)
    transferred_net.summary()
    transferred_net.add(K.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=activation))
    transferred_net.add(K.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=activation))
    transferred_net.add(K.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=activation))

    transferred_net.add(K.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation=activation))
    transferred_net.add(K.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation=activation))
    transferred_net.add(K.layers.SpatialDropout2D(rate=dropout_rate))

    transferred_net.add(K.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='same', activation=activation))
    transferred_net.add(K.layers.Conv2D(filters=1, kernel_size=[3, 3], padding='same', activation='sigmoid'))

    transferred_net.layers[-1].name = OUT(0)
    return transferred_net


def transfer_mobile_net_joints(dropout_rate=0.0, activation=K.layers.activations.relu, train_mobilenet=False):

    # net_in = K.layers.Input(shape=[224, 224, 3], name=IN('img'))
    # print(net_in._op.__dict__)

    mobile_net = MobileNet(include_top=False,
                           weights='imagenet',
                           dropout=dropout_rate,
                           # input_tensor=net_in._op,
                           input_shape=[224, 224, 3])
    layers = mobile_net.layers[:26]
    for layer in layers:
        layer.trainable = train_mobilenet
    layers[0].name = IN('img')

    transferred_net = K.models.Sequential(layers=layers)
    c1 = K.layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation=activation)(transferred_net.output)
    c1 = K.layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation=activation)(c1)
    c1 = K.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=activation)(c1)

    c2 = K.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=activation)(c1)
    c2 = K.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=activation)(c2)
    d2 = K.layers.SpatialDropout2D(rate=dropout_rate)(c2)

    c3 = K.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='valid', activation=activation)(d2)
    c3 = K.layers.Conv2D(filters=21, kernel_size=[3, 3], padding='valid', activation='sigmoid', name=OUT('heats'))(c3)

    # before the fully connected is built, cut down the dimensionality of the data
    bfc1 = K.layers.MaxPool2D(padding='valid', pool_size=(2, 2))(d2)
    bfc2 = K.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='valid', activation=activation)(bfc1)
    base_fc = K.layers.Flatten()(bfc2)
    fc1 = K.layers.Dense(units=32, activation=activation)(base_fc)
    fc2 = K.layers.Dense(units=21, activation='sigmoid', name=OUT('vis'))(fc1)

    model = km.Model(inputs=(transferred_net.input,), outputs=(c3, fc2))
    print("############### SHAPE " + str(model.output_shape))
    return model


if __name__ == "__main__":
    net = transfer_mobile_net(False)
    net.summary()
