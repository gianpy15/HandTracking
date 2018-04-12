from neural_network.keras.models.heatmap import *
from neural_network.keras.callbacks.image_writer import ImageWriter
from neural_network.keras.custom_layers.heatmap_loss import my_loss
from tensorboard_utils.tensorboard_manager import TensorBoardManager as TBManager
from data_manager.path_manager import resources_path
from neural_network.keras.utils.data_loader import *
import os

DEFAULT_TENSORBOARD_PATH = "heat_maps"
DEFAULT_CHECKPOINT_PATH = resources_path(os.path.join('models/hand_cropper/cropper_v5.ckp'))
DEFAULT_H5MODEL_PATH = resources_path(os.path.join('models/hand_cropper/cropper_v5.h5'))


def train_model(model, dataset,
                tb_path=None, checkpoint_path=None, h5model_path=None,
                learning_rate=1e-3, batch_size=10, epochs=50, patience=5):

    if tb_path is None:
        tb_path = resources_path(os.path.join("tbdata", DEFAULT_TENSORBOARD_PATH))

    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT_PATH

    if h5model_path is None:
        h5model_path = DEFAULT_H5MODEL_PATH

    TBManager.set_path(tb_path)
    tb_manager_train = TBManager()
    tb_manager_test = TBManager()

    tb_manager_test.add_images(dataset[VALID_IN][0:5], name="test_imgs", max_out=5)
    tb_manager_test.add_images(dataset[VALID_TARGET][0:5], name="test_maps", max_out=5)
    tb_manager_train.add_images(dataset[TRAIN_IN][0:5], name="train_imgs", max_out=5)
    tb_manager_train.add_images(dataset[TRAIN_TARGET][0:5], name="train_maps", max_out=5)

    # Callbacks for keras
    tensor_board = kc.TensorBoard(log_dir=tb_path, histogram_freq=1)
    model_ckp = kc.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss',
                                   verbose=1, save_best_only=True, mode='min', period=1)
    es = kc.EarlyStopping(patience=patience, verbose=1, monitor='val_loss', mode='min', min_delta=2e-4)
    im = ImageWriter(images=dataset[TRAIN_IN][0:5], tb_manager=tb_manager_train, name='train_output')
    im2 = ImageWriter(images=dataset[VALID_IN][0:5], tb_manager=tb_manager_test, name='test_output')
    callbacks = [tensor_board, model_ckp, es, im, im2]

    # Training tools
    optimizer = ko.adam(lr=learning_rate)
    loss = my_loss
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # tb_manager.clear_data()

    print('training starting...')
    model.fit(dataset[TRAIN_IN], dataset[TRAIN_TARGET], epochs=epochs,
              batch_size=batch_size, callbacks=callbacks, verbose=1,
              validation_data=(dataset[VALID_IN], dataset[VALID_TARGET]))
    print('training complete!')

    model.save(h5model_path)
    print("Model saved")
    return model
