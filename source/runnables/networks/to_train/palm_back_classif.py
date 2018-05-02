import sys
import os

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..", "..")))

from library.neural_network.keras.models.palm_back_classifier import *
from data.datasets.palm_back_classifier.pb_classifier_ds_management import *
from keras.models import load_model
import keras.backend as k
import data.regularization.regularizer as reg
import numpy as np
import keras

# palm visible (+1.0)
# back visible (-1.0)

name = "palm_back_classifier_unseq.h5"
batch_size = 2
path = resources_path("palm_back_classification_dataset_rgb")

if __name__ == '__main__':

    LOAD_MODEL = False

    TRAIN_MODEL = True

    regularizer = reg.Regularizer()
    regularizer.fixresize(200, 200)

    #create_dataset(savepath=path, im_regularizer=regularizer)

    x_train, y_train, c_train, x_test, y_test, c_test = read_dataset(path=path, leave_out=['handsMaddalena2',
                                                                                'handsGianpy',
                                                                                'handsMatteo'], minconf=0.999)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    c_train = np.array(c_train)

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    c_test = np.array(c_test)
    print(x_train.shape, y_train.shape, c_train.shape, x_test.shape, y_test.shape, c_test.shape)

    class_weight = count_ones_zeros(y_train, y_test)
    print(class_weight)

   # x_train, y_train, c_train = shuffle_cut_label_conf(x_train, y_train, c_train)
    #x_test, y_test, c_test = shuffle_cut_label_conf(x_test, y_test, c_test)

    if LOAD_MODEL:
        # change the name of the model to be loaded
        model = load_model(resources_path(os.path.join("models", "palm_back", name)))
    else:
        model = unsequential_model()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    if TRAIN_MODEL:
        model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=2, verbose=1, class_weight=class_weight,
                  validation_data=(x_test, y_test))

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