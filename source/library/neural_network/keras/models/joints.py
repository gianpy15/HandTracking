from data import *
from library.neural_network.keras.models.building_blocks import *


def uniform_model(kernel, num_filters):
    model = km.Sequential()

    model.add(kl.Conv2D(filters=num_filters,
                        kernel_size=kernel,
                        activation='relu',
                        input_shape=(None, None, 3),
                        data_format="channels_last",
                        padding="same"
                        )
              )

    model.add(kl.Conv2D(filters=num_filters,
                        kernel_size=kernel,
                        activation='relu',
                        data_format="channels_last"
                        )
              )

    model.add(kl.Conv2D(filters=num_filters,
                        kernel_size=kernel,
                        activation='relu',
                        data_format="channels_last"
                        )
              )
    model.add(kl.Conv2D(filters=num_filters,
                        kernel_size=kernel,
                        activation='relu',
                        data_format="channels_last"
                        )
              )

    model.add(kl.Conv2D(filters=num_filters,
                        kernel_size=kernel,
                        activation='relu',
                        data_format="channels_last"
                        )
              )

    model.add(kl.Conv2D(filters=num_filters,
                        kernel_size=kernel,
                        activation='relu',
                        data_format="channels_last"
                        )
              )

    model.add(kl.Conv2D(filters=num_filters,
                        kernel_size=kernel,
                        activation='relu',
                        data_format="channels_last"
                        )
              )

    model.add(kl.Conv2D(filters=num_filters,
                        kernel_size=kernel,
                        activation='relu',
                        data_format="channels_last"
                        )
              )

    model.add(kl.Conv2D(filters=num_filters,
                        kernel_size=kernel,
                        activation='relu',
                        data_format="channels_last"
                        )
              )

    model.add(kl.Conv2D(filters=num_filters,
                        kernel_size=kernel,
                        activation='relu',
                        data_format="channels_last"
                        )
              )

    model.add(kl.Conv2D(filters=num_filters,
                        kernel_size=kernel,
                        activation='relu',
                        data_format="channels_last"
                        )
              )

    model.add(kl.Conv2D(filters=num_filters,
                        kernel_size=kernel,
                        activation='relu',
                        data_format="channels_last"
                        )
              )

    # model.add(kl.MaxPooling2D(pool_size=pool))

    return model


def high_fov_model(weight_decay=0):
    model = km.Sequential()
    model.add(kl.Conv2D(input_shape=(None, None, 3), filters=32, kernel_size=[7, 7], padding='same'))
    model.add(kl.Activation('relu'))
    # FOV: 7
    model.add(kl.Conv2D(filters=64, kernel_size=[7, 7], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    # FOV: 13
    model.add(kl.Conv2D(filters=128, kernel_size=[5, 5], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    # FOV: 17
    model.add(kl.Conv2D(filters=256, kernel_size=[5, 5], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    # FOV: 21
    model.add(kl.MaxPooling2D())
    # FOV: 42
    model.add(kl.Conv2D(filters=128, kernel_size=[5, 5], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    # FOV: 50
    model.add(kl.Conv2D(filters=64, kernel_size=[5, 5], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    # FOV: 58
    model.add(kl.Conv2D(filters=21, kernel_size=[3, 3], padding='same', activation='relu',
                        kernel_regularizer=weight_decay))
    # FOV: 62
    model.add(kl.Activation('sigmoid'))

    return model


def low_injection_locator(channels=3, name='refiner', dropout_rate=0, activation='relu'):
    inputs = kl.Input(shape=(None, None, channels), name=IN('img'))

    in1 = kl.BatchNormalization()(inputs)

    # LOW FEATURES BLOCK
    block1 = parallel_conv_block(in_tensor=in1,
                                 end_ch=256,
                                 activation=activation,
                                 kernel_size=[5, 5],
                                 topology=(2, 4, 6))  # ========\
    pool1 = kl.MaxPool2D(pool_size=[2, 2])(block1)  # .         \\ GRADIENT
    normblock1 = kl.BatchNormalization()(pool1)     # .          \\
    # LOW LEVEL GRADIENT INJECTION                               // INJECTION
    branch1 = serial_pool_reduction_block(in_tensor=block1,  # <=/
                                          end_ch=64,
                                          topology=(2, 2),
                                          activation=activation,
                                          kernel_size=[3, 3])
    heats1 = kl.Conv2D(filters=21, activation='sigmoid', kernel_size=[3, 3], padding='same', name=OUT('mid_heats'))(branch1)

    # MID FEATURES BLOCK
    block2 = parallel_conv_block(in_tensor=normblock1,
                                 end_ch=512,
                                 activation=activation,
                                 kernel_size=[3, 3],
                                 topology=(8, 3))
    block2 = kl.Dropout(rate=dropout_rate)(block2)

    conv1 = kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same')(block2)
    act1 = kl.Activation(activation)(conv1) if type(activation) is str else activation()(conv1)
    conv1 = kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same')(act1)
    act1 = kl.Activation(activation)(conv1) if type(activation) is str else activation()(conv1)
    act1 = kl.BatchNormalization()(act1)
    pool2 = kl.MaxPool2D(pool_size=[2, 2])(act1)

    # OUTPUT DIMENSIONS REACHED - HIGH LEVEL FEATURES
    # reduce number of features before dense visibility prediction
    conv2 = kl.Conv2D(filters=32, kernel_size=[3, 3], padding='same')(pool2)
    act2 = kl.Activation(activation)(conv1) if type(activation) is str else activation()(conv2)
    out_vis = kl.Flatten()(act2)
    out_vis = kl.Dense(activation='sigmoid', use_bias=True, units=21, name=OUT('vis'))(out_vis)
    extended_conv2 = kl.concatenate(inputs=[conv2, pool2])
    heats2 = kl.Conv2D(filters=21, activation='sigmoid', kernel_size=[5, 5], padding='same', name=OUT('heats'))(extended_conv2)

    model = km.Model(inputs=[inputs], outputs=[heats1, heats2, out_vis], name=name)

    return model