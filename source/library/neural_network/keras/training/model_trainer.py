import os

import keras as K
import keras.callbacks as kc
import keras.optimizers as ko

from data.naming import *
from library.neural_network.keras.callbacks.image_writer import ImageWriter
from library.neural_network.keras.custom_layers.heatmap_loss import prop_heatmap_loss
from library.neural_network.tensorboard_interface.tensorboard_manager import TensorBoardManager as TBManager
from library.telegram.telegram_bot import *
from library.utils.visualization_utils import get_image_with_mask


def train_model(model_generator, dataset, loss=prop_heatmap_loss,
                tb_path='', model_name=None, model_type=None,
                learning_rate=1e-3, batch_size=10, epochs=50, patience=-1,
                additional_callbacks=None, verbose=False):
    K.backend.clear_session()

    model = model_generator()

    if model_name is None or model_type is None:
        checkpoint_path = None
        h5model_path = None
    else:
        if tb_path is not None:
            os.path.join(tb_path, model_name)
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
                                     name='valid_output'))
    if additional_callbacks is not None:
        callbacks += additional_callbacks

    # Training tools
    optimizer = ko.adam(lr=learning_rate)

    if verbose:
        print("Compiling model...")

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
    if verbose:
        print('Fitting model...')
        # Notification for telegram
        try:
            notify_training_starting(model_name=model_type + "_" + model_name,
                                     training_samples=len(dataset[TRAIN_IN]),
                                     validation_samples=len(dataset[VALID_IN]),
                                     tensorboard="handtracking.eastus.cloudapp.azure.com:6006 if active")
        except Exception:
            pass

    history = model.fit(dataset[TRAIN_IN], dataset[TRAIN_TARGET], epochs=epochs,
                        batch_size=batch_size, callbacks=callbacks, verbose=1,
                        validation_data=(dataset[VALID_IN], dataset[VALID_TARGET]))
    if verbose:
        print('Fitting completed!')
        loss_ = "{:.5f}".format(history.history['loss'][-1])
        valid_loss = "{:.5f}".format(history.history['val_loss'][-1])
        accuracy = "{:.2f}%".format(100 * history.history['acc'][-1])
        valid_accuracy = "{:.2f}%".format(100 * history.history['val_acc'][-1])
        try:
            notify_training_end(model_name=model_type + "_" + model_name,
                                final_loss=str(loss_),
                                final_validation_loss=str(valid_loss),
                                final_accuracy=str(accuracy),
                                final_validation_accuracy=str(valid_accuracy),
                                tensorboard="handtracking.eastus.cloudapp.azure.com:6006 if active")
        except Exception:
            pass

        if model_name == CROPPER:
            try:
                send_message("Training sample...")
                img = dataset[TRAIN_IN][0]
                map_ = model.predict(img)
                send_image_from_array(get_image_with_mask(img, map_))
                send_image_from_array(get_image_with_mask(img, map_))
                send_message("Validation sample...")
                img = dataset[VALID_IN][0]
                map_ = model.predict(img)
                send_image_from_array(get_image_with_mask(img, map_))
                send_image_from_array(get_image_with_mask(img, map_))
            except Exception:
                pass

    if h5model_path is not None:
        if verbose:
            print("Saving H5 model...")
        model = model_generator()
        model.load_weights(checkpoint_path)
        model.save(h5model_path)
        os.remove(checkpoint_path)

    if verbose:
        print("Training completed!")

    return model
