from neural_network.keras.models.heatmap import *
from neural_network.keras.callbacks.image_writer import ImageWriter
from neural_network.keras.custom_layers.heatmap_loss import prop_heatmap_loss
from tensorboard_utils.tensorboard_manager import TensorBoardManager as TBManager
from neural_network.keras.utils.naming import *
import keras as K
import os

DEFAULT_CHECKPOINT_PATH = resources_path(os.path.join('models/hand_cropper/cropper_v5.ckp'))
DEFAULT_H5MODEL_PATH = resources_path(os.path.join('models/hand_cropper/cropper_v5.h5'))


def train_model(model_generator, dataset,
                tb_path='', model_name=None, model_type=None,
                learning_rate=1e-3, batch_size=10, epochs=50, patience=-1,
                additional_callbacks=None, loss_white_prio=-1.5, verbose=False):
    K.backend.clear_session()

    model = model_generator()

    if model_name is None or model_type is None:
        checkpoint_path = None
        h5model_path = None
    else:
        if model_type == CROPPER:
            checkpoint_path = cropper_ckp_path(model_name)
            h5model_path = cropper_h5_path(model_name)
        elif model_type == JLOCATOR:
            checkpoint_path = jlocator_ckp_path(model_name)
            h5model_path = jlocator_h5_path(model_name)
        else:
            checkpoint_path = None
            h5model_path = None

    if verbose:
        print("Model:")
        model.summary()

    callbacks = []
    if checkpoint_path is not None:
        if verbose:
            print("Adding callback for checkpoint...")
        callbacks.append(kc.ModelCheckpoint(filepath=checkpoint_path,
                                            monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='min',
                                            period=1))
    if patience > 0:
        if verbose:
            print("Adding callback for early stopping...")
        callbacks.append(kc.EarlyStopping(patience=patience,
                                          verbose=1,
                                          monitor='val_loss',
                                          mode='min',
                                          min_delta=2e-4))

    if tb_path is not None:
        TBManager.set_path(tb_path)
        if verbose:
            print("Setting up tensorboard...")
            print("Clearing tensorboard files...")
        TBManager.clear_data()

        if verbose:
            print("Adding tensorboard callbacks...")
        callbacks.append(kc.TensorBoard(log_dir=tensorboard_path(tb_path),
                                        histogram_freq=1))
        callbacks.append(ImageWriter(data=(dataset[TRAIN_IN],
                                           dataset[TRAIN_TARGET]),
                                     name='train_output'))
        callbacks.append(ImageWriter(data=(dataset[VALID_IN],
                                           dataset[VALID_TARGET]),
                                     name='test_output'))
    if additional_callbacks is not None:
        callbacks += additional_callbacks

    # Training tools
    optimizer = ko.adam(lr=learning_rate)

    if verbose:
        print("Compiling model...")

    model.compile(optimizer=optimizer,
                  loss=lambda hgr, hpr: prop_heatmap_loss(hgr, hpr, white_priority=loss_white_prio),
                  metrics=['accuracy'])
    if verbose:
        print('Fitting model...')
    model.fit(dataset[TRAIN_IN], dataset[TRAIN_TARGET], epochs=epochs,
              batch_size=batch_size, callbacks=callbacks, verbose=1,
              validation_data=(dataset[VALID_IN], dataset[VALID_TARGET]))
    if verbose:
        print('Fitting completed!')

    if h5model_path is not None:
        if verbose:
            print("Saving H5 model...")
        model.save(h5model_path)

    if verbose:
        print("Training completed!")

    return model
