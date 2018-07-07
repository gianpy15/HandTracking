from data import *
from library.neural_network.keras.models.building_blocks import *
import keras.layers as kl
import keras.models as km
from keras.applications.mobilenet import MobileNet


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


def low_injection_locator(input_shape, name='refiner', dropout_rate=0, activation='relu'):
    inputs = kl.Input(shape=input_shape, name=IN('img'))

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


def finger_field_injection(dropout_rate=0.0, activation=kl.activations.relu, train_mobilenet=False):

    # net_in = kl.Input(shape=[224, 224, 3], name=IN('img'))
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

    transferred_net = km.Sequential(layers=layers)

    # alternative path from input
    c_alt = kl.Conv2D(filters=64, kernel_size=[7, 7], padding='same', activation=activation)(transferred_net.input)
    c_alt = kl.Conv2D(filters=256, kernel_size=[5, 5], padding='same', activation=activation)(c_alt)
    c_alt = kl.MaxPool2D(padding='same', pool_size=(2, 2))(c_alt)

    c_alt = kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=activation)(c_alt)
    c_alt = kl.Conv2D(filters=transferred_net.output_shape[-1], kernel_size=[3, 3], padding='same', activation=activation)(c_alt)
    c_alt = kl.MaxPool2D(padding='same', pool_size=(2, 2))(c_alt)

    base_in = kl.concatenate([transferred_net.output, c_alt], axis=-1)
    c1 = kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation=activation)(base_in)
    c1 = kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation=activation)(c1)
    c1 = kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=activation)(c1)

    c_vec = kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation=activation)(base_in)
    c_vec = kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=activation)(c_vec)
    c_vec_out = kl.Conv2D(filters=10, kernel_size=[3, 3], padding='same', activation='tanh', name=OUT('field'))(c_vec)

    c_vec = kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=activation)(c_vec_out)

    c2_in = kl.concatenate([c1, c_vec], axis=-1)

    c2 = kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=activation)(c2_in)
    c2 = kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=activation)(c2)
    d2 = kl.SpatialDropout2D(rate=dropout_rate)(c2)

    c3 = kl.Conv2D(filters=64, kernel_size=[3, 3], padding='valid', activation=activation)(d2)
    c3 = kl.Conv2D(filters=21, kernel_size=[3, 3], padding='valid', activation='sigmoid', name=OUT('heat'))(c3)

    # before the fully connected is built, cut down the dimensionality of the data
    bfc1 = kl.MaxPool2D(padding='valid', pool_size=(2, 2))(d2)
    bfc2 = kl.Conv2D(filters=32, kernel_size=[3, 3], padding='valid', activation=activation)(bfc1)
    base_fc = kl.Flatten()(bfc2)
    fc1 = kl.Dense(units=32, activation=activation)(base_fc)
    fc2 = kl.Dense(units=21, activation='sigmoid', name=OUT('vis'))(fc1)

    model = km.Model(inputs=(transferred_net.input,), outputs=(c3, fc2, c_vec_out))
    return model
