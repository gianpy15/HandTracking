import keras.layers as klayers
import keras.models as kmod

def simple_classifier():

    model = kmod.Sequential()

    model.add(klayers.Conv2D(input_shape=(200, 200, 1), filters=70, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=90, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=100, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=120, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=120, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=150, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=150, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=150, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=150, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=150, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=250, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=200, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=150, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Flatten())

    model.add(klayers.Dense(units=100, activation='relu', use_bias=True))

    model.add(klayers.Dense(units=100, activation='relu', use_bias=True))

    model.add(klayers.Dense(units=100, activation='relu', use_bias=True))

    # model.add(klayers.Dropout(rate=0.1, seed=12345))

    model.add(klayers.Dense(units=1, activation='sigmoid'))

    return model


def simple_classifier_rgb():

    model = kmod.Sequential()

    model.add(klayers.Conv2D(input_shape=(200, 200, 3), filters=100, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=150, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=200, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=250, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=250, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=200, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=200, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=150, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=125, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=100, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Flatten())

    model.add(klayers.Dense(units=50, activation='relu', use_bias=True))

    model.add(klayers.Dense(units=80, activation='relu', use_bias=True))

    # model.add(klayers.Dropout(rate=0.1, seed=12345))

    model.add(klayers.Dense(units=1, activation='sigmoid'))

    return model


def simple_classifier_rgb2():

    model = kmod.Sequential()

    model.add(klayers.Conv2D(input_shape=(200, 200, 3), filters=80, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=130, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=130, kernel_size=[1, 1], activation='relu'))

    model.add(klayers.Conv2D(filters=130, kernel_size=[1, 1], activation='relu'))

    model.add(klayers.Conv2D(filters=130, kernel_size=[1, 1], activation='relu'))

    model.add(klayers.Conv2D(filters=130, kernel_size=[1, 1], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=130, kernel_size=[1, 1], activation='relu'))

    model.add(klayers.Conv2D(filters=150, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=150, kernel_size=[1, 1], activation='relu'))

    model.add(klayers.Conv2D(filters=150, kernel_size=[1, 1], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=160, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=170, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=200, kernel_size=[1, 1], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=200, kernel_size=[1, 1], activation='relu'))

    model.add(klayers.Conv2D(filters=200, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=200, kernel_size=[1, 1], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=200, kernel_size=[3, 3], activation='relu'))

    model.add(klayers.Conv2D(filters=200, kernel_size=[1, 1], activation='relu'))

    model.add(klayers.Conv2D(filters=250, kernel_size=[2, 2], activation='relu'))

    model.add(klayers.Conv2D(filters=250, kernel_size=[1, 1], activation='relu'))

    model.add(klayers.MaxPooling2D(pool_size=[2, 2]))

    model.add(klayers.Conv2D(filters=250, kernel_size=[2, 2], activation='relu'))

    model.add(klayers.Conv2D(filters=250, kernel_size=[1, 1], activation='relu'))

    model.add(klayers.Conv2D(filters=250, kernel_size=[2, 2], activation='relu'))

    model.add(klayers.Conv2D(filters=250, kernel_size=[1, 1], activation='relu'))

    model.add(klayers.Conv2D(filters=500, kernel_size=[1, 1], activation='relu'))

    model.add(klayers.Flatten())

    model.add(klayers.Dense(units=100, activation='relu', use_bias=True))

    model.add(klayers.Dense(units=80, activation='linear', use_bias=True))

    model.add(klayers.Dense(units=80, activation='relu', use_bias=True))

    # model.add(klayers.Dropout(rate=0.1, seed=12345))

    model.add(klayers.Dense(units=1, activation='sigmoid'))

    return model


def unsequential_model():
    inputs = kmod.Input(shape=(200, 200, 3))
    x = klayers.Conv2D(filters=60, kernel_size=[3, 3], padding='same')(inputs)
    x = klayers.Conv2D(filters=3, kernel_size=[3, 3], padding='same')(x)

    x = klayers.Add()([x, inputs])

    x = klayers.Conv2D(filters=120, kernel_size=[3, 3], activation='relu')(x)
    x = klayers.Conv2D(filters=170, kernel_size=[3, 3], activation='relu')(x)
    x = klayers.MaxPool2D(pool_size=[2, 2])(x)
    x = klayers.Conv2D(filters=200, kernel_size=[3, 3], activation='relu')(x)
    x = klayers.Conv2D(filters=200, kernel_size=[3, 3], activation='relu')(x)
    x = klayers.Conv2D(filters=200, kernel_size=[1, 1], activation='relu')(x)
    x = klayers.MaxPool2D(pool_size=[2, 2])(x)
    x = klayers.Conv2D(filters=200, kernel_size=[3, 3], activation='relu')(x)
    x = klayers.Conv2D(filters=200, kernel_size=[1, 1], activation='relu')(x)
    x = klayers.Conv2D(filters=150, kernel_size=[3, 3], activation='relu')(x)
    x = klayers.MaxPool2D(pool_size=[2, 2])(x)
    x = klayers.Conv2D(filters=120, kernel_size=[3, 3], activation='relu')(x)
    x = klayers.Conv2D(filters=100, kernel_size=[1, 1], activation='relu')(x)
    x = klayers.MaxPool2D(pool_size=[2, 2])(x)
    x = klayers.Conv2D(filters=200, kernel_size=[2, 2], activation='relu')(x)
    x = klayers.Conv2D(filters=250, kernel_size=[2, 2], activation='relu')(x)
    x = klayers.MaxPool2D(pool_size=[2, 2])(x)
    x = klayers.Conv2D(filters=250, kernel_size=[2, 2], activation='relu')(x)
    x = klayers.Conv2D(filters=300, kernel_size=[1, 1], activation='relu')(x)
    x = klayers.Flatten()(x)
    x = klayers.Dense(units=50, activation='relu', use_bias=True)(x)
    x = klayers.Dense(units=20, activation='linear', use_bias=True)(x)
    x = klayers.Dense(units=1, activation='sigmoid', use_bias=True)(x)
    return kmod.Model(inputs=[inputs], outputs=x)


def unsequential_model_heat():
    inputs = kmod.Input(shape=(200, 200, 3))
    x = klayers.Conv2D(filters=60, kernel_size=[3, 3], padding='same')(inputs)
    x = klayers.Conv2D(filters=30, kernel_size=[3, 3], padding='same')(x)
    x = klayers.Conv2D(filters=3, kernel_size=[3, 3], padding='same')(x)

    x = klayers.Add()([x, inputs])
    x = klayers.Conv2D(filters=150, kernel_size=[3, 3])(x)
    x = klayers.Conv2D(filters=200, kernel_size=[3, 3])(x)
    x = klayers.MaxPool2D(pool_size=[2, 2])(x)
    x = klayers.Conv2D(filters=200, kernel_size=[2, 2])(x)
    x = klayers.Conv2D(filters=250, kernel_size=[2, 2])(x)
    x = klayers.MaxPool2D(pool_size=[2, 2])(x)
    x = klayers.Conv2D(filters=250, kernel_size=[2, 2])(x)
    x = klayers.Conv2D(filters=250, kernel_size=[2, 2])(x)
    x = klayers.MaxPool2D(pool_size=[2, 2])(x)
    x = klayers.Conv2D(filters=250, kernel_size=[2, 2])(x)
    x = klayers.Conv2D(filters=250, kernel_size=[2, 2])(x)
    x = klayers.MaxPool2D(pool_size=[2, 2])(x)
    x = klayers.Conv2D(filters=250, kernel_size=[2, 2])(x)
    x = klayers.Conv2D(filters=200, kernel_size=[2, 2])(x)
    x = klayers.Conv2D(filters=150, kernel_size=[3, 3])(x)

    x = klayers.Flatten()(x)
    x = klayers.Dense(units=50, activation='relu', use_bias=True)(x)
    x = klayers.Dense(units=50, activation='relu', use_bias=True)(x)
    x = klayers.Dense(units=1, activation='sigmoid', use_bias=True)(x)
    return kmod.Model(inputs=inputs, outputs=x)