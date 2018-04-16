from neural_network.keras.models.right_left_classifier import simple_classifier
from left_right_classifier_utils.left_right_classif_ds_management import *
from keras.models import load_model
import hands_regularizer.regularizer as reg
import numpy as np
import keras

if __name__ == '__main__':

    LOAD_MODEL = True

    TRAIN_MODEL = False

    #regularizer = reg.Regularizer()
    #regularizer.fixresize(200, 200)
    #regularizer.rgb2gray()

    #create_dataset(im_regularizer=regularizer)

    x_train, y_train, x_test, y_test = read_dataset(leave_out=['handsMaddalena2', 'handsGianpy', 'handsMatteo'])
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    count_ones_zeros(y_train, y_test)

    x_train, y_train = shuffle_cut_label(x_train, y_train)
    x_test, y_test = shuffle_cut_label(x_test, y_test)

    if LOAD_MODEL:
        # change the name of the model to be loaded
        model = load_model("right_left_classifier.h5")
    else:
        model = simple_classifier()


    if TRAIN_MODEL:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x=x_train, y=y_train, batch_size=20, epochs=5, verbose=1)

        # change the name of the model to save
        model.save("right_left_classifier.h5")

    evaluation_score = model.evaluate(x=x_test, y=y_test, batch_size=20, verbose=1)

    print("Model evaluation:", evaluation_score)

    n = x_test.shape[0]
    ris = model.predict(x_test)
    correct = 0
    wrong = 0
    for i in range(n):
        if y_test[i] == 1 and ris[i] >= 0.5 or y_test[i] == 0 and ris[i] < 0.5:
            correct += 1
        else:
            wrong += 1
            print("Expected: ", y_test[i], "|  Predicted: ", ris[i])
    print("Correct: ", correct)
    print("Wrong: ", wrong)


