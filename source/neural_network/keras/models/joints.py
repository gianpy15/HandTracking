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
