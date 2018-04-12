from keras.optimizers import Adam
from neural_network.keras.custom_layers.heatmap_loss import my_loss
import numpy as np
import junctions_locator_utils.junction_locator_ds_management as jlocator
import hands_regularizer.regularizer as regularizer
from neural_network.keras.models.joints import *
from neural_network.keras.utils.model_trainer import train_model
from neural_network.keras.utils.naming import *

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

    BUILD_DATASET = True
    VERBOSE = True

    # input building

    img_reg = regularizer.Regularizer()
    img_reg.fixresize(resize, resize)
    hm_reg = regularizer.Regularizer()
    hm_reg.fixresize(hm_resize, hm_resize)
    hm_reg.heatmaps_threshold(threshold)
    if BUILD_DATASET:
        jlocator.create_dataset(["handsBorgo2"], im_regularizer=img_reg, heat_regularizer=hm_reg, enlarge=.5, cross_radius=5)
    cuts, hms, visible = jlocator.read_dataset(verbosity=1)
    # Note:
    #   not it holds:
    #   hms.shape == (frames, 21, hm_resize, hm_resize, 3)
    #   but should be:
    #   hms.shape == (frames, hm_resize, hm_resize, 21)
    # probably need bugfix on jlocator.read_dataset
    x_train = np.array(cuts[0:1])
    y_train = np.array(hms[0:1])

    if VERBOSE:
        print("train set shape: " + str(x_train.shape))
        print("target shape: " + str(y_train.shape))

    # Models are collected in neural_network/keras/models
    model = high_fov_model(weight_decay=kl.regularizers.l2(1e-5))
    # config

    model = train_model(model=model,
                        dataset={TRAIN_IN: x_train,
                                 TRAIN_TARGET: y_train,
                                 VALID_IN: x_train,
                                 VALID_TARGET: y_train},
                        epochs=1,
                        tb_path=None,
                        patience=5,
                        verbose=True)
