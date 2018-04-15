from keras import backend as K


def reshape_image(batch, width, height):
    if K.image_data_format() == 'channels_first':
        x = batch.reshape(batch.shape[0], 1, width, height)
    else:
        x = batch.reshape(batch.shape[0], width, height, 1)

    if x.dtype not in ['float16', 'float32', 'float64']:
        x = x.astype('float32')
        x /= 255
    return x
