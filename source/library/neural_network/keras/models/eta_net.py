import keras as K
import keras.layers as kl
from keras.models import Model
from data import IN, OUT


def eta_net(input_shape, weight_decay=None, name='u_net', dropout_rate=0, activation='relu'):
    inputs = kl.Input(shape=input_shape, name=IN(0))

    # Encoding part of the network
    conv1 = kl.Conv2D(filters=32, kernel_size=[5, 5], padding='same', kernel_regularizer=weight_decay)(inputs)
    act1 = kl.Activation(activation)(conv1) if type(activation) is str else activation()(conv1)
    conv1 = kl.Conv2D(filters=32, kernel_size=[5, 5], padding='same', kernel_regularizer=weight_decay)(act1)
    act1 = kl.Activation(activation)(conv1) if type(activation) is str else activation()(conv1)
    pool1 = kl.MaxPool2D(pool_size=[2, 2])(act1)

    conv2 = kl.Conv2D(filters=64, kernel_size=[5, 5], padding='same', kernel_regularizer=weight_decay)(pool1)
    act2 = kl.Activation(activation)(conv2) if type(activation) is str else activation()(conv2)
    conv2 = kl.Conv2D(filters=64, kernel_size=[5, 5], padding='same', kernel_regularizer=weight_decay)(act2)
    act2 = kl.Activation(activation)(conv2) if type(activation) is str else activation()(conv2)
    pool2 = kl.MaxPool2D(pool_size=[2, 2])(act2)

    conv3 = kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', kernel_regularizer=weight_decay)(pool2)
    act3 = kl.Activation(activation)(conv3) if type(activation) is str else activation()(conv3)
    conv3 = kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', kernel_regularizer=weight_decay)(act3)
    act3 = kl.Activation(activation)(conv3) if type(activation) is str else activation()(conv3)
    norm3 = kl.BatchNormalization()(act3)
    pool3 = kl.MaxPool2D(pool_size=[2, 2])(norm3)

    conv4 = kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same', kernel_regularizer=weight_decay)(pool3)
    act4 = kl.Activation(activation)(conv4) if type(activation) is str else activation()(conv4)
    conv4 = kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same', kernel_regularizer=weight_decay)(act4)
    act4 = kl.Activation(activation)(conv4) if type(activation) is str else activation()(conv4)
    norm4 = kl.BatchNormalization()(act4)
    drop4 = kl.Dropout(rate=dropout_rate)(norm4)
    pool4 = kl.MaxPool2D(pool_size=[2, 2])(drop4)

    conv5 = kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same', kernel_regularizer=weight_decay)(pool4)
    act5 = kl.Activation(activation)(conv5) if type(activation) is str else activation()(conv5)
    conv5 = kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same', kernel_regularizer=weight_decay)(act5)
    act5 = kl.Activation(activation)(conv5) if type(activation) is str else activation()(conv5)
    drop5 = kl.Dropout(rate=dropout_rate)(act5)

    # Decoding part of the network
    up6 = kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same',
                    kernel_regularizer=weight_decay)(kl.UpSampling2D(size=[2, 2])(drop5))
    act6 = kl.Activation(activation)(up6) if type(activation) is str else activation()(up6)
    norm6 = kl.BatchNormalization()(act6)
    merge6 = kl.concatenate(inputs=[norm4, norm6])
    conv6 = kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same', kernel_regularizer=weight_decay)(merge6)
    act6 = kl.Activation(activation)(conv6) if type(activation) is str else activation()(conv6)
    conv6 = kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same', kernel_regularizer=weight_decay)(act6)
    act6 = kl.Activation(activation)(conv6) if type(activation) is str else activation()(conv6)

    up7 = kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same',
                    kernel_regularizer=weight_decay)(kl.UpSampling2D(size=[2, 2])(act6))
    act7 = kl.Activation(activation)(up7) if type(activation) is str else activation()(up7)
    norm7 = kl.BatchNormalization()(act7)
    merge7 = kl.concatenate(inputs=[norm3, norm7])
    conv7 = kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', kernel_regularizer=weight_decay)(merge7)
    act7 = kl.Activation(activation)(conv7) if type(activation) is str else activation()(conv7)
    conv7 = kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', kernel_regularizer=weight_decay)(act7)
    act7 = kl.Activation(activation)(conv7) if type(activation) is str else activation()(conv7)
    drop7 = kl.Dropout(rate=dropout_rate)(act7)

    # Output with one filter and sigmoid activation function
    out = kl.Conv2D(filters=1, kernel_size=[1, 1], activation='sigmoid', kernel_regularizer=weight_decay,
                    name=OUT(0))(drop7)

    eta_net_model = Model(inputs=(inputs,), outputs=(out,), name=name)

    return eta_net_model


if __name__ == '__main__':
    model = eta_net(input_shape=[512, 512, 3], activation=lambda: K.layers.LeakyReLU(alpha=0.1))
    model.summary()
