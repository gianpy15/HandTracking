import keras.layers as klayers
import keras.models as kmod

def simple_classifier():

    model = kmod.Sequential()

    model.add(klayers.Conv2D(input_shape=(200, 200, 3), filters=5, kernel_size=[3, 3], padding='same', activation='relu'))

    model.add(klayers.Conv2D(filters=5, kernel_size=[3, 3], padding='same', activation='relu'))

    model.add(klayers.Conv2D(filters=5, kernel_size=[3, 3], padding='same', activation='relu'))

    model.add(klayers.MaxPooling2D())

    model.add(klayers.Conv2D(filters=10, kernel_size=[3, 3], padding='same', activation='relu'))

    model.add(klayers.Conv2D(filters=10, kernel_size=[3, 3], padding='same', activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[3, 3]))

    model.add(klayers.Flatten())

    model.add(klayers.Dense(units=15, activation='relu', use_bias=True))

    model.add(klayers.Dense(units=1, activation='sigmoid'))

    return model



