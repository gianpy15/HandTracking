import keras.layers as klayers
import keras.models as kmod
from keras import backend as k
from data import IN, OUT

def l1_reg(weight_matrix):
    return 0.01 * k.sum(k.abs(weight_matrix))


def l2_reg(weight_matrix):
    return 0.01 * k.sum(k.square(weight_matrix))


def simple_classifier_rgb(weight_decay=None):
    inputs = klayers.Input(shape=(200, 200, 3), name=IN(0))

    x = klayers.Conv2D(filters=70, kernel_size=[3, 3], activation='relu')(inputs)

    x = klayers.Conv2D(filters=80, kernel_size=[3, 3], activation='relu')(x)

    x = klayers.Conv2D(filters=90, kernel_size=[3, 3], activation='relu')(x)

    x = klayers.MaxPooling2D(pool_size=[2, 2])(x)

    x = klayers.Conv2D(filters=90, kernel_size=[3, 3], activation='relu')(x)

    x = klayers.Conv2D(filters=100, kernel_size=[3, 3], activation='relu')(x)

    x = klayers.Conv2D(filters=120, kernel_size=[3, 3], activation='relu')(x)

    x = klayers.MaxPooling2D(pool_size=[2, 2])(x)

    x = klayers.Conv2D(filters=130, kernel_size=[3, 3], activation='relu')(x)

    x = klayers.Conv2D(filters=150, kernel_size=[3, 3], activation='relu')(x)

    x = klayers.MaxPooling2D(pool_size=[2, 2])(x)

    x = klayers.Conv2D(filters=170, kernel_size=[3, 3], activation='relu')(x)

    x = klayers.Conv2D(filters=180, kernel_size=[3, 3], activation='relu')(x)

    x = klayers.MaxPooling2D(pool_size=[2, 2])(x)

    x = klayers.Conv2D(filters=190, kernel_size=[3, 3], activation='relu')(x)

    x = klayers.Conv2D(filters=200, kernel_size=[3, 3], activation='relu')(x)

    x = klayers.Flatten()(x)

    x = klayers.Dense(units=100, activation='relu', use_bias=True)(x)

    x = klayers.Dropout(rate=0.15)(x)

    x = klayers.Dense(units=1, activation='sigmoid', name=OUT(0))(x)

    return kmod.Model(inputs=(inputs,), outputs=(x,))

