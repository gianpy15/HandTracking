# In this file i'll create a model that given a binary sequence
# says if that sequence contains or not a 1

import keras.models as km
import keras.layers as kl
import keras.callbacks as kc
import numpy as np
from random import *

# Preparing the data
n_samples = 500
elements_for_sample = 100
probability_of_one = 0.5
x_train = np.zeros([n_samples, elements_for_sample])
y_train = np.zeros([n_samples, 1])
x_test = np.zeros([n_samples, elements_for_sample])
y_test = np.zeros([n_samples, 1])

for i in range(n_samples):
    if random() < probability_of_one:
        x_train[i, randint(0, elements_for_sample - 1)] = 1
        y_train[i, 0] = 1
        x_test[i, randint(0, elements_for_sample - 1)] = 1
        y_test[i, 0] = 1

# Create the model
model = km.Sequential()
model.add(kl.Embedding(input_dim=2, output_dim=64))
model.add(kl.LSTM(units=1, dropout=0.2, recurrent_dropout=0.2))
model.add(kl.Dense(1, activation='sigmoid'))

outputs = [layer.output for layer in model.layers]

tensor_board = kc.TensorBoard(histogram_freq=1)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
print('training starting...')
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test),
                    callbacks=[tensor_board], verbose=2)
print('training complete!')
