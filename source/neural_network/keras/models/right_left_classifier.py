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



