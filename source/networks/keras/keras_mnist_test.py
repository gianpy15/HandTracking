from keras.datasets import mnist
import keras.models as km
import keras.layers as kl
import keras.callbacks as kc
import keras
from networks.keras.utils import reshape_image

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:5000]
y_train = y_train[:5000]

tb_dir = '../../../resources/tensorboard_utils/tbdata'

x_train = reshape_image(x_train, 28, 28)
x_test = reshape_image(x_test, 28, 28)


print('x_train shape:', x_train.shape)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = km.Sequential()
model.add(kl.Flatten(input_shape=x_train.shape[1:]))
model.add(kl.Dense(50))
model.add(kl.Activation('relu'))
model.add(kl.Dense(50))
model.add(kl.Activation('relu'))
model.add(kl.Dropout(0.4))
model.add(kl.Dense(10, activation='softmax'))

tensor_board = kc.TensorBoard(histogram_freq=1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, epochs=50, callbacks=[tensor_board],
                    validation_data=(x_test, y_test), verbose=2)
