import keras.layers as klayers
import keras.models as kmod

def simple_classifier():

    model = kmod.Sequential()

    model.add(klayers.Conv2D(input_shape=(200, 200, 1), filters=20, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=30, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=40, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=50, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=70, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=80, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=100, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=250, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Flatten())

    model.add(klayers.Dense(units=100, activation='relu', use_bias=True))

    model.add(klayers.Dropout(rate=0.1, seed=12345))

    model.add(klayers.Dense(units=1, activation='sigmoid'))

    return model


def simple_classifier2():

    model = kmod.Sequential()

    model.add(klayers.Conv2D(input_shape=(200, 200, 1), filters=50, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=70, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=90, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=100, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=120, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=150, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=150, kernel_size=[3, 3], activation='relu', padding='same'))

    model.add(klayers.Conv2D(filters=150, kernel_size=[3, 3], activation='relu', padding='same'))

    model.add(klayers.Conv2D(filters=150, kernel_size=[3, 3], activation='relu', padding='same'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=170, kernel_size=[3, 3], activation='relu', padding='same'))

    model.add(klayers.Conv2D(filters=170, kernel_size=[3, 3], activation='relu', padding='same'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=175, kernel_size=[3, 3], activation='relu', padding='same'))

    model.add(klayers.Conv2D(filters=200, kernel_size=[3, 3], activation='relu', padding='same'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=256, kernel_size=[3, 3], activation='relu', padding='same'))

    model.add(klayers.Conv2D(filters=256, kernel_size=[3, 3], activation='relu', padding='same'))

    model.add(klayers.Conv2D(filters=256, kernel_size=[3, 3], activation='relu', padding='same'))

    model.add(klayers.Flatten())

    model.add(klayers.Dense(units=50, activation='relu', use_bias=True))

    model.add(klayers.Dense(units=50, activation='relu', use_bias=True))

    model.add(klayers.Dense(units=50, activation='relu', use_bias=True))

    model.add(klayers.Dropout(rate=0.1, seed=12345))

    model.add(klayers.Dense(units=1, activation='sigmoid'))

    return model


def simple_classifier3():

    model = kmod.Sequential()

    model.add(klayers.Conv2D(input_shape=(150, 150, 1), filters=100, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=130, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=150, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=200, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=250, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=250, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=200, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=100, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=75, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=60, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=45, kernel_size=[3, 3], activation='relu'))


    model.add(klayers.Flatten())

    model.add(klayers.Dense(units=60, activation='relu', use_bias=True))

    model.add(klayers.Dense(units=60, activation='relu', use_bias=True))

    model.add(klayers.Dense(units=60, activation='relu', use_bias=True))

    #model.add(klayers.Dropout(rate=0.1, seed=12345))

    model.add(klayers.Dense(units=1, activation='sigmoid'))

    return model


def complex_classifier():

    model = kmod.Sequential()

    model.add(klayers.Conv2D(input_shape=(200, 200, 1), filters=60, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=70, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=80, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=90, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=100, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=120, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=150, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=180, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=190, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=200, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=210, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=220, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=220, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=220, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=220, kernel_size=[3, 3], activation='relu', padding='same'))

    model.add(klayers.Conv2D(filters=220, kernel_size=[3, 3], activation='relu', padding='same'))

    model.add(klayers.Conv2D(filters=250, kernel_size=[3, 3], activation='relu', padding='same'))

    model.add(klayers.Flatten())

    model.add(klayers.Dense(units=100, activation='relu', use_bias=True))

    model.add(klayers.Dropout(rate=0.05, seed=12345))

    model.add(klayers.Dense(units=1, activation='sigmoid'))

    return model


