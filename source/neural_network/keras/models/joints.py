import keras.models as km
import keras.layers as kl


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
