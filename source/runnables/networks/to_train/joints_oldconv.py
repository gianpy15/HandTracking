import sys
import os

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..", "..")))
from keras.utils.generic_utils import get_custom_objects

from library.neural_network.keras.models.joints import *
from data.datasets.jlocator.junction_locator_ds_management import *
from keras.models import load_model
import keras.regularizers as kr
import keras.optimizers as ko
import numpy as np
import keras.backend as kb


def loss(y_true, y_pred):
    tp = kb.relu(y_true - y_pred) * 10
    pt = kb.relu(y_pred - y_true)
    return kb.sum(tp + pt)


name = "joints_test2.h5"
batch_size = 5
weight_decay = kr.l2(1e-5) # not used yet
learning_rate = 1e-3

path = joints_path()

if __name__ == '__main__':

    LOAD_MODEL = False
    TRAIN_MODEL = True

    x_train, y_train, v_train, x_test, y_test, v_test = read_dataset(path=path,
                                                                     test_vids=['handsMaddalena2', 'handsMatteo'])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    v_train = np.array(v_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    v_test = np.array(v_test)
    print(x_train.shape, y_train.shape, v_train.shape, x_test.shape, y_test.shape, v_test.shape)

    # x_train, y_train, c_train = shuffle_cut_label_conf(x_train, y_train, c_train)
    # x_test, y_test, c_test = shuffle_cut_label_conf(x_test, y_test, c_test)
    if LOAD_MODEL:

        model = load_model(resources_path(os.path.join("models", "joints", name)))
    else:
        model = uniform_unseq_model([3, 3], 200)
    model.compile(optimizer=ko.adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    if TRAIN_MODEL:
        model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=2, verbose=1,
                  validation_data=(x_test, y_test))
        # change the name of the model to save
        model.save(resources_path(os.path.join("models", "joints", name)))

    evaluation_score = model.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=1)
    print("Model evaluation:", evaluation_score)
    n = x_test.shape[0]
    ris = model.predict(x_test, batch_size=batch_size)

    u.showimage(x_test[0])
    for i in range(21):
        u.showimage(y_test[0][:, :, i])
        u.showimage(ris[0][:, :, i])
