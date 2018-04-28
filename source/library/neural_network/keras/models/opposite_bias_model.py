from library.neural_network.keras.custom_layers.heatmap_loss import *
from library.neural_network.keras.custom_layers.abs import Abs


def opposite_bias_adversarial(channels=3, weight_decay=None, name='opposite_bias_adversarial', activation='relu'):
    model = km.Sequential(name=name)
    model.add(kl.Conv2D(input_shape=(None, None, channels), filters=64, kernel_size=[7, 7], padding='same'))
    model.add(kl.Activation(activation) if type(activation) is str else activation())
    # model.add(kl.BatchNormalization())

    model.add(kl.Conv2D(filters=128, kernel_size=[5, 5], padding='same',
                        kernel_regularizer=weight_decay))
    model.add(kl.Activation(activation) if type(activation) is str else activation())
    # model.add(kl.BatchNormalization())

    model.add(kl.Conv2D(filters=256, kernel_size=[5, 5], padding='same',
                        kernel_regularizer=weight_decay))
    model.add(kl.Activation(activation) if type(activation) is str else activation())
    model.add(kl.BatchNormalization())
    model.add(kl.MaxPooling2D())

    model.add(kl.Conv2D(filters=256, kernel_size=[5, 5], padding='same',
                        kernel_regularizer=weight_decay))
    model.add(kl.Activation(activation) if type(activation) is str else activation())
    # model.add(kl.BatchNormalization())

    model.add(kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same',
                        kernel_regularizer=weight_decay))
    model.add(kl.Activation(activation) if type(activation) is str else activation())
    model.add(kl.BatchNormalization())
    model.add(kl.MaxPooling2D())

    model.add(kl.Conv2D(filters=64, kernel_size=[3, 3], padding='same',
                        kernel_regularizer=weight_decay))
    model.add(kl.Activation(activation) if type(activation) is str else activation())
    # model.add(kl.BatchNormalization())

    model.add(kl.Conv2D(filters=32, kernel_size=[3, 3], padding='same',
                        kernel_regularizer=weight_decay))
    model.add(kl.Activation(activation) if type(activation) is str else activation())
    # model.add(kl.BatchNormalization())

    model.add(kl.Conv2D(filters=1, kernel_size=[5, 5], padding='same', kernel_regularizer=weight_decay))
    model.add(kl.Activation('sigmoid'))

    return model


def opposite_bias_regularizer(channels=5, weight_decay=None, name='opposite_bias_regularizer', activation='relu'):
    model = km.Sequential(name=name)
    model.add(kl.Conv2D(input_shape=(None, None, channels), filters=64, kernel_size=[5, 5], padding='same'))
    model.add(kl.Activation(activation) if type(activation) is str else activation())
    model.add(kl.BatchNormalization())

    model.add(kl.Conv2D(filters=128, kernel_size=[5, 5], padding='same',
                        kernel_regularizer=weight_decay))
    model.add(kl.Activation(activation) if type(activation) is str else activation())
    model.add(kl.BatchNormalization())

    model.add(kl.Conv2D(filters=256, kernel_size=[5, 5], padding='same',
                        kernel_regularizer=weight_decay))
    model.add(kl.Activation(activation) if type(activation) is str else activation())
    model.add(kl.BatchNormalization())
    model.add(kl.MaxPooling2D())

    model.add(kl.Conv2D(filters=128, kernel_size=[5, 5], padding='same',
                        kernel_regularizer=weight_decay))
    model.add(kl.Activation(activation) if type(activation) is str else activation())
    model.add(kl.BatchNormalization())

    model.add(kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same',
                        kernel_regularizer=weight_decay))
    model.add(kl.Activation(activation) if type(activation) is str else activation())
    model.add(kl.BatchNormalization())

    model.add(kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same',
                        kernel_regularizer=weight_decay))
    model.add(kl.Activation(activation) if type(activation) is str else activation())
    model.add(kl.BatchNormalization())
    model.add(kl.MaxPooling2D())

    model.add(kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same',
                        kernel_regularizer=weight_decay))
    model.add(kl.Activation(activation) if type(activation) is str else activation())
    model.add(kl.BatchNormalization())

    model.add(kl.Conv2D(filters=64, kernel_size=[3, 3], padding='same',
                        kernel_regularizer=weight_decay))
    model.add(kl.Activation(activation) if type(activation) is str else activation())
    model.add(kl.BatchNormalization())

    model.add(kl.Conv2D(filters=32, kernel_size=[3, 3], padding='same',
                        kernel_regularizer=weight_decay))
    model.add(kl.Activation(activation) if type(activation) is str else activation())
    model.add(kl.BatchNormalization())

    model.add(kl.Conv2D(filters=1, kernel_size=[3, 3], padding='same', kernel_regularizer=weight_decay))
    model.add(kl.Activation('sigmoid'))

    return model


if __name__ == '__main__':
    model = opposite_bias_adversarial(3, activation=Abs)
    model.summary()
