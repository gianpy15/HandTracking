import sys
import os

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..", "..")))

from library.neural_network.keras.models.palm_back_classifier import *
from data.datasets.palm_back_classifier.pb_classifier_ds_management import *
from data.datasets.crop import utils as u
from keras.models import load_model
import keras.backend as k
import data.regularization.regularizer as reg
import numpy as np
import keras

# palm visible (+1.0)
# back visible (-1.0)

name = "palm_back_classifier_unseq_h.h5"
batch_size = 2
path = resources_path("palm_back_classification_dataset_h")


def prod_im_heat(im, heat):
    heat = np.dstack((heat, heat, heat))
    return np.array(np.multiply(im, heat), dtype=np.uint8)


def prod_im_heat_batch(ims, heats):
    n = len(ims)
    ris = []
    for i in range(n):
        prod = prod_im_heat(ims[i], heats[i])
        ris.append(prod)
    return ris



if __name__ == '__main__':

    LOAD_MODEL = False

    TRAIN_MODEL = True

    regularizer = reg.Regularizer()
    regularizer.fixresize(200, 200)
    regularizer_h = reg.Regularizer()
    regularizer_h.fixresize(200, 200)

    #create_dataset_w_heatmaps(savepath=path, im_regularizer=regularizer, h_r=regularizer_h)

    x_train, y_train, c_train, h_train, x_test, y_test, c_test, h_test = read_dataset_h(path=path,
                                                                                        leave_out=['handsMaddalena2',
                                                                                                    'handsGianpy',
                                                                                                    'handsMatteo'])
    x_train = prod_im_heat_batch(x_train, h_train)
    x_test = prod_im_heat_batch(x_test, h_test)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    c_train = np.array(c_train)
    h_train = np.array(h_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    c_test = np.array(c_test)
    h_test = np.array(h_test)

    count_ones_zeros(y_train, y_test)

    class_weight = {0: 531/194, 1: 531/337}

    x_train, y_train, c_train, h_train = shuffle_cut_label_conf_h(x_train, y_train, c_train, h_train)
    x_test, y_test, c_test, h_test = shuffle_cut_label_conf_h(x_test, y_test, c_test, h_test)

    if LOAD_MODEL:
        # change the name of the model to be loaded
        model = load_model(resources_path(os.path.join("models", "palm_back", name)))
    else:
        model = unsequential_model_heat()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    if TRAIN_MODEL:
        model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=2, verbose=1, class_weight=class_weight)

        # change the name of the model to save
        model.save(resources_path(os.path.join("models", "palm_back", name)))

    evaluation_score = model.evaluate(x=x_test, y=y_test, batch_size=batch_size, verbose=1)

    print("Model evaluation:", evaluation_score)

    n = x_test.shape[0]
    ris = model.predict(x_test, batch_size=batch_size)
    correct = 0
    wrong = 0
    for i in range(n):

        if y_test[i] == 1 and ris[i] >= 0.5 or y_test[i] == 0 and ris[i] < 0.5:
            correct += 1
        else:
            #if wrong < 10:
                #u.showimage(x_test[i].squeeze())
            wrong += 1
        print("Expected: ", y_test[i], "|  Predicted: ", ris[i])
    print("Correct: ", correct)
    print("Wrong: ", wrong)
