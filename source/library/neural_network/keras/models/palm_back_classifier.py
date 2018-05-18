import keras.layers as klayers
import keras.models as kmod
from keras import backend as k
from data import IN, OUT

def l1_reg(weight_matrix):
    return 0.01 * k.sum(k.abs(weight_matrix))


def l2_reg(weight_matrix):
    return 0.01 * k.sum(k.square(weight_matrix))


def simple_classifier_rgb(weight_decay=None):
    model = kmod.Sequential()

    model.add(klayers.Conv2D(input_shape=(200, 200, 3), filters=50, kernel_size=[3, 3], activation='relu', name=IN(0)))

    model.add(klayers.Conv2D(filters=60, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=60, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=70, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=80, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=90, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=100, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=150, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=170, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=180, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=200, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Flatten())

    model.add(klayers.Dense(units=100, activation='relu', use_bias=True))

    #model.add(klayers.Dropout(rate=0.1, seed=12345))

    model.add(klayers.Dense(units=1, activation='sigmoid', name=OUT(0)))

    return model

