from keras.optimizers import Adam
from neural_network.keras.custom_layers.heatmap_loss import my_loss
import numpy as np
import junctions_locator_utils.junction_locator_ds_management as jlocator
import hands_regularizer.regularizer as regularizer
from neural_network.keras.models.joints import *

if __name__ == '__main__':
    # some values

    resize = 200
    hm_resize = 100
    input_height = resize
    input_width = resize
    threshold = .5
    batch_size = 25
    num_filters = 15
    kernel = (3, 3)
    pool = (2, 2)

    BUILD_DATASET = False
    VERBOSE = True

    # input building

    img_reg = regularizer.Regularizer()
    img_reg.fixresize(resize, resize)
    hm_reg = regularizer.Regularizer()
    hm_reg.fixresize(hm_resize, hm_resize)
    hm_reg.heatmaps_threshold(threshold)
    if BUILD_DATASET:
        jlocator.create_dataset(["handsAlberto1"], im_regularizer=img_reg, heat_regularizer=hm_reg, enlarge=.5, cross_radius=5)
    cuts, hms, visible = jlocator.read_dataset(verbosity=1)
    # Note:
    #   not it holds:
    #   hms.shape == (frames, 21, hm_resize, hm_resize, 3)
    #   but should be:
    #   hms.shape == (frames, hm_resize, hm_resize, 21)
    # probably need bugfix on jlocator.read_dataset
    x_train = np.array(cuts)
    y_train = np.array(hms)

    if VERBOSE:
        print("train set shape: " + str(x_train.shape))
        print("target shape: " + str(y_train.shape))

    # Models are collected in neural_network/keras/models
    model = uniform_model(kernel=kernel, num_filters=num_filters)
    # config

    model.compile(
        optimizer=Adam(),
        loss=my_loss,
        metrics=['accuracy']
    )

    # training

    model.fit(epochs=100, batch_size=batch_size, x=x_train, y=y_train)
