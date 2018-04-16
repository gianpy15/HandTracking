from neural_network.keras.models.right_left_classifier import simple_classifier
from left_right_classifier_utils.left_right_classif_ds_management import create_dataset
from left_right_classifier_utils.left_right_classif_ds_management import read_dataset
from left_right_classifier_utils.left_right_classif_ds_management import  read_dataset_random
from keras.models import load_model
import hands_regularizer.regularizer as reg
import numpy as np


if __name__ == '__main__':

    LOAD_MODEL = False

    regularizer = reg.Regularizer()
    regularizer.fixresize(200, 200)

    create_dataset(["handsA", "handsAlberto1"], im_regularizer=regularizer)

    x_train, y_train, x_test, y_test = read_dataset()

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_test = np.array(x_test)
    y_test = np.array(y_test)


    if LOAD_MODEL:
        # change the name of the model to be loaded
        model = load_model("right_left_classifier.h5")
    else:
        model = simple_classifier()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x=x_train, y=y_train, batch_size=20, epochs=100, verbose=1)

    # change the name of the model to save
    model.save("right_left_classifier.h5")

    evaluation_score = model.evaluate(x=x_test, y=y_test, batch_size=20, verbose=1)

    print("Model evaluation:"+evaluation_score)


