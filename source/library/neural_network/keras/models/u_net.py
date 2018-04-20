import keras.layers as kl
import keras.models as km


def u_net(input_shape, weight_decay=None, name='u_net', dropout_rate=0):
    inputs = kl.Input(shape=input_shape)

    # Encoding part of the network
    conv1 = kl.Conv2D(filters=64, kernel_size=[3, 3], activation='relu', padding='same')(inputs)
    conv1 = kl.Conv2D(filters=64, kernel_size=[3, 3], activation='relu', padding='same')(conv1)
    pool1 = kl.MaxPool2D(pool_size=[2, 2])(conv1)

    conv2 = kl.Conv2D(filters=128, kernel_size=[3, 3], activation='relu', padding='same')(pool1)
    conv2 = kl.Conv2D(filters=128, kernel_size=[3, 3], activation='relu', padding='same')(conv2)
    pool2 = kl.MaxPool2D(pool_size=[2, 2])(conv2)

    conv3 = kl.Conv2D(filters=256, kernel_size=[3, 3], activation='relu', padding='same')(pool2)
    conv3 = kl.Conv2D(filters=256, kernel_size=[3, 3], activation='relu', padding='same')(conv3)
    pool3 = kl.MaxPool2D(pool_size=[2, 2])(conv3)

    conv4 = kl.Conv2D(filters=512, kernel_size=[3, 3], activation='relu', padding='same')(pool3)
    conv4 = kl.Conv2D(filters=512, kernel_size=[3, 3], activation='relu', padding='same')(conv4)
    drop4 = kl.Dropout(rate=dropout_rate)(conv4)
    pool4 = kl.MaxPool2D(pool_size=[2, 2])(drop4)

    conv5 = kl.Conv2D(filters=1024, kernel_size=[3, 3], activation='relu', padding='same')(pool4)
    conv5 = kl.Conv2D(filters=1024, kernel_size=[3, 3], activation='relu', padding='same')(conv5)
    drop5 = kl.Dropout(rate=dropout_rate)(conv5)

    # Decoding part of the network
    up6 = kl.Conv2D(filters=512, kernel_size=[3, 3], activation='relu', padding='same') \
        (kl.UpSampling2D(size=[2, 2])(drop5))
    merge6 = kl.merge(inputs=[drop4, up6], mode='concat', concat_axis=-1)
    conv6 = kl.Conv2D(filters=512, kernel_size=[3, 3], activation='relu', padding='same')(merge6)
    conv6 = kl.Conv2D(filters=512, kernel_size=[3, 3], activation='relu', padding='same')(conv6)

    up7 = kl.Conv2D(filters=256, kernel_size=[3, 3], activation='relu', padding='same') \
        (kl.UpSampling2D(size=[2, 2])(conv6))
    merge7 = kl.merge(inputs=[conv3, up7], mode='concat', concat_axis=-1)
    conv7 = kl.Conv2D(filters=256, kernel_size=[3, 3], activation='relu', padding='same')(merge7)
    conv7 = kl.Conv2D(filters=256, kernel_size=[3, 3], activation='relu', padding='same')(conv7)

    up8 = kl.Conv2D(filters=128, kernel_size=[3, 3], activation='relu', padding='same') \
        (kl.UpSampling2D(size=[2, 2])(conv7))
    merge8 = kl.merge(inputs=[conv2, up8], mode='concat', concat_axis=-1)
    conv8 = kl.Conv2D(filters=128, kernel_size=[3, 3], activation='relu', padding='same')(merge8)
    conv8 = kl.Conv2D(filters=128, kernel_size=[3, 3], activation='relu', padding='same')(conv8)

    up9 = kl.Conv2D(filters=64, kernel_size=[3, 3], activation='relu', padding='same') \
        (kl.UpSampling2D(size=[2, 2])(conv8))
    merge9 = kl.merge(inputs=[conv1, up9], mode='concat', concat_axis=-1)
    conv9 = kl.Conv2D(filters=64, kernel_size=[3, 3], activation='relu', padding='same')(merge9)
    conv9 = kl.Conv2D(filters=64, kernel_size=[3, 3], activation='relu', padding='same')(conv9)

    # Output with one filter and sigmoid activation function
    out = kl.Conv2D(filters=1, kernel_size=[1, 1], activation='sigmoid')(conv9)

    u_net_model = km.Model(input=inputs, output=out, name=name)

    return u_net_model


if __name__ == '__main__':
    model = u_net(input_shape=[512, 512, 3])
    model.summary()
