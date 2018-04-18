from library.neural_network.keras.custom_layers.heatmap_loss import *


def opposite_bias_adversarial(channels=3, weight_decay=None, name='opposite_bias_adversarial'):
    model = km.Sequential(name=name)
    model.add(kl.Conv2D(input_shape=(None, None, channels), filters=64, kernel_size=[5, 5], padding='same'))
    model.add(kl.Activation('relu'))
    model.add(kl.Conv2D(filters=128, kernel_size=[5, 5], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.Conv2D(filters=256, kernel_size=[5, 5], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.MaxPooling2D())
    model.add(kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.MaxPooling2D())
    model.add(kl.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.Conv2D(filters=32, kernel_size=[3, 3], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.Conv2D(filters=1, kernel_size=[3, 3], padding='same', kernel_regularizer=weight_decay))
    model.add(kl.Activation('sigmoid'))

    return model


def opposite_bias_regularizer(channels=5, weight_decay=None, name='opposite_bias_regularizer'):
    model = km.Sequential(name=name)
    model.add(kl.Conv2D(input_shape=(None, None, channels), filters=64, kernel_size=[5, 5], padding='same'))
    model.add(kl.Activation('relu'))
    model.add(kl.Conv2D(filters=128, kernel_size=[5, 5], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.Conv2D(filters=256, kernel_size=[5, 5], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.MaxPooling2D())
    model.add(kl.Conv2D(filters=128, kernel_size=[5, 5], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.MaxPooling2D())
    model.add(kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.Conv2D(filters=32, kernel_size=[3, 3], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    model.add(kl.Conv2D(filters=1, kernel_size=[3, 3], padding='same', kernel_regularizer=weight_decay))
    model.add(kl.Activation('sigmoid'))

    return model
