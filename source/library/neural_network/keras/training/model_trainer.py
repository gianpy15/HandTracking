import keras as K
import keras.callbacks as kc
import keras.optimizers as ko
import traceback

from data import *
from library import *
from library.neural_network.keras.callbacks.image_writer import ImageWriter
from library.neural_network.keras.callbacks.scalar_writer import ScalarWriter
from library.neural_network.keras.custom_layers.heatmap_loss import prop_heatmap_loss
from library.neural_network.keras.sequence import BatchGenerator
from library.neural_network.tensorboard_interface.tensorboard_manager import TensorBoardManager as TBManager
from library.telegram import telegram_bot as tele
from library.utils.visualization_utils import get_image_with_mask
from library.neural_network.batch_processing.processing_plan import ProcessingPlan


def train_model(model_generator, dataset_manager: DatasetManager, loss=prop_heatmap_loss,
                tb_path='', model_name=None, model_type=None, data_processing_plan: ProcessingPlan=None,
                learning_rate=1e-3, epochs=50, patience=-1,
                additional_callbacks=None, verbose=False):
    K.backend.clear_session()

    train_data = dataset_manager.train()
    valid_data = dataset_manager.valid()

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
    log("Model:", level=COMMENTARY)
    model.summary(print_fn=lambda s: log(s, level=COMMENTARY))

    callbacks = []
    if checkpoint_path is not None:
        log("Adding callback for checkpoint...", level=COMMENTARY)
        callbacks.append(kc.ModelCheckpoint(filepath=checkpoint_path,
                                            monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True,
                                            mode='min',
                                            period=1))
    if patience > 0:
        log("Adding callback for early stopping...", level=COMMENTARY)
        callbacks.append(kc.EarlyStopping(patience=patience,
                                          verbose=1,
                                          monitor='val_loss',
                                          mode='min',
                                          min_delta=2e-4))

    if tb_path is not None:
        TBManager.set_path(tb_path)
        log("Setting up tensorboard...", level=COMMENTARY)
        log("Clearing tensorboard files...", level=COMMENTARY)
        TBManager.clear_data()

        log("Adding tensorboard callbacks...", level=COMMENTARY)
        callbacks.append(ScalarWriter())
        callbacks.append(ImageWriter(data=(train_data[0][IN(0)],
                                           train_data[0][OUT(0)]),
                                     name='train_output'))
        callbacks.append(ImageWriter(data=(valid_data[0][IN(0)],
                                           valid_data[0][OUT(0)]),
                                     name='valid_output', freq=3))
    if additional_callbacks is not None:
        callbacks += additional_callbacks

    # Training tools
    optimizer = ko.adam(lr=learning_rate)

    log("Compiling model...", level=COMMENTARY)

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])

    if verbose:
        log('Fitting model...', level=COMMENTARY)
        # Notification for telegram
        try:
            tele.notify_training_starting(model_name=model_type + "_" + model_name,
                                          training_samples=len(train_data) * dataset_manager.batch_size,
                                          validation_samples=len(valid_data) * dataset_manager.batch_size,
                                          tensorboard="handtracking.eastus.cloudapp.azure.com:6006 if active")
        except Exception:
            traceback.print_exc()

    history = model.fit_generator(generator=BatchGenerator(data_sequence=train_data,
                                                           process_plan=data_processing_plan),
                                  epochs=epochs, verbose=1, callbacks=callbacks,
                                  validation_data=BatchGenerator(data_sequence=valid_data,
                                                                 process_plan=data_processing_plan))

    if h5model_path is not None:
        log("Saving H5 model...", level=COMMENTARY)
        model = model_generator()
        model.load_weights(checkpoint_path)
        model.save(h5model_path)
        os.remove(checkpoint_path)

    log("Training completed!", level=COMMENTARY)

    if verbose:
        log('Fitting completed!', level=COMMENTARY)
        loss_ = "{:.5f}".format(history.history['loss'][-1])
        valid_loss = "{:.5f}".format(history.history['val_loss'][-1])
        accuracy = "{:.2f}%".format(100 * history.history['acc'][-1])
        valid_accuracy = "{:.2f}%".format(100 * history.history['val_acc'][-1])
        try:
            tele.notify_training_end(model_name=model_type + "_" + model_name,
                                     final_loss=str(loss_),
                                     final_validation_loss=str(valid_loss),
                                     final_accuracy=str(accuracy),
                                     final_validation_accuracy=str(valid_accuracy))

            if model_type == CROPPER:
                tele.send_message(message="Training samples:")
                img = train_data[0][IN(0)] * 255
                map_ = model.predict(img)
                tele.send_image_from_array(get_image_with_mask(img, map_))
                tele.send_message(message="Validation samples:")
                img = valid_data[0][IN(0)] * 255
                map_ = model.predict(img)
                tele.send_image_from_array(get_image_with_mask(img, map_))
        except Exception:
            traceback.print_exc()

    return model
