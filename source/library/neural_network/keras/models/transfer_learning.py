import keras as K
from data import *


def transfer_vgg(weight_decay=None, dropout_rate=0.0, activation=K.layers.activations.relu, train_vgg=False):
    vgg = K.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=None)
    layers = vgg.layers[:10]
    for layer in layers:
        layer.trainable = train_vgg
    layers[0].name = IN(0)
    transferred_net = K.models.Sequential(layers=layers)
    transferred_net.add(K.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same',
                                        activity_regularizer=weight_decay,
                                        kernel_regularizer=weight_decay, activation=activation))
    transferred_net.add(K.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same',
                                        activity_regularizer=weight_decay,
                                        kernel_regularizer=weight_decay, activation=activation))
    transferred_net.add(K.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same',
                                        activity_regularizer=weight_decay,
                                        kernel_regularizer=weight_decay, activation=activation))

    transferred_net.add(K.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same',
                                        activity_regularizer=weight_decay,
                                        kernel_regularizer=weight_decay, activation=activation))
    transferred_net.add(K.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same',
                                        activity_regularizer=weight_decay,
                                        kernel_regularizer=weight_decay, activation=activation))
    transferred_net.add(K.layers.SpatialDropout2D(rate=dropout_rate))

    transferred_net.add(K.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='same',
                                        activity_regularizer=weight_decay,
                                        kernel_regularizer=weight_decay, activation=activation))
    transferred_net.add(K.layers.Conv2D(filters=1, kernel_size=[3, 3], padding='same',
                                        kernel_regularizer=weight_decay, activation='sigmoid'))

    transferred_net.layers[-1].name = OUT(0)

    return transferred_net


if __name__ == "__main__":
    net = transfer_vgg()
    net.summary()
