import os
import sys

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "../..")))

from neural_network.keras.models.heatmap import *
from neural_network.keras.callbacks.image_writer import ImageWriter
from neural_network.keras.callbacks.input_writer import InputWriter
from neural_network.keras.custom_layers.heatmap_loss import my_loss
from tensorboard_utils.tensorboard_manager import TensorBoardManager as TBManager
from keras.engine import training as kt
from skimage.transform import rescale
from keras import backend as K
from neural_network.keras.utils.data_loader import *


dataset_path = resources_path(os.path.join("hands_bounding_dataset", "network_test"))
tensorboard_path = resources_path(os.path.join("tbdata/heat_maps"))
model_ck_path = resources_path(os.path.join('models/hand_cropper/incremental/cropper_v5'))
model_save_path = resources_path(os.path.join('models/hand_cropper/incremental/cropper_v5'))

TBManager.set_path("heat_maps")
tb_manager_train = TBManager('train_images')
tb_manager_test = TBManager('test_images')
train = True

# Hyper parameters
weight_decay = kr.l2(1e-5)
learning_rate = 1e-3

dataset = load_dataset(train_samples=2,
                       valid_samples=2,
                       dataset_path=dataset_path,
                       random_dataset=True,
                       shuffle=True,
                       use_depth=False,
                       verbose=True)
tb_manager_test.add_images(dataset[VALID_IN][0:5], name="test_imgs", max_out=5)
tb_manager_test.add_images(dataset[VALID_TARGET][0:5], name="test_maps", max_out=5)
tb_manager_train.add_images(dataset[TRAIN_IN][0:5], name="train_imgs", max_out=5)
tb_manager_train.add_images(dataset[TRAIN_TARGET][0:5], name="train_maps", max_out=5)


def fit_new_model(_4d_inputs, _4d_tests, decay=0, train_model=False, model_type=incremental_predictor_1, name=""):
    # Training tools
    K.clear_session()
    model = model_type(input_shape=np.shape(_4d_inputs)[1:], weight_decay=decay, name=name)
    model.summary()
    optimizer = ko.adam(lr=learning_rate)
    loss = my_loss
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    if train_model:
        # Callbacks for keras
        tensor_board = kc.TensorBoard(log_dir=tensorboard_path + "/" + name, histogram_freq=1)
        es = kc.EarlyStopping(patience=5, verbose=1, monitor='val_loss', mode='min', min_delta=2e-4)
        im = ImageWriter(images=dataset[TRAIN_IN][0:5], tb_manager=tb_manager_train, name='train_output')
        im2 = ImageWriter(images=dataset[VALID_IN][0:5], tb_manager=tb_manager_test, name='test_output')
        in_w = InputWriter(name="heatmap_input")
        model_ckp = kc.ModelCheckpoint(filepath=model_ck_path + name + ".ckp", monitor='val_loss',
                                       verbose=1, save_best_only=True, mode='min', period=1)
        callbacks = [es, model_ckp, tensor_board]
        print("train shapes: {} {}".format(np.shape(_4d_inputs), np.shape(_4d_tests)))
        model.fit(_4d_inputs, dataset[TRAIN_TARGET], epochs=2, batch_size=1, verbose=1, callbacks=callbacks,
                  validation_data=(_4d_tests, dataset[VALID_TARGET]))
        model.save(model_save_path + name + ".h5")
        print("Saved model " + name)

    return model


def attach_heat_map(inputs, fitted_model: kt.Model):
    _inputs = inputs[:, :, :, 0:3]
    outputs = fitted_model.predict(inputs)
    rescaled = []
    for img in outputs:
        rescaled.append(rescale(img, 4.0))
    outputs = np.array(rescaled)
    inputs_ = np.concatenate((_inputs, outputs), axis=-1)
    return inputs_


# Build up the model
# First model part
new_inputs = np.concatenate((dataset[TRAIN_IN], np.zeros(shape=np.shape(dataset[TRAIN_IN])[0:-1] + (1,))), axis=-1)
new_tests = np.concatenate((dataset[VALID_IN], np.zeros(shape=np.shape(dataset[VALID_IN])[0:-1] + (1,))), axis=-1)
print(np.shape(dataset[VALID_IN]))
model1 = fit_new_model(new_inputs, new_tests, weight_decay, True, name="_m1")

new_inputs = attach_heat_map(new_inputs, model1)
new_tests = attach_heat_map(new_tests, model1)
print("****input shape**** {}".format(np.shape(new_inputs)))
print("****test shape**** {}".format(np.shape(new_tests)))
# Second Model
model2 = fit_new_model(new_inputs, new_tests, weight_decay, True, name="_m2")

new_inputs = attach_heat_map(new_inputs, model2)
new_tests = attach_heat_map(new_tests, model2)
# Third Model
model3 = fit_new_model(new_inputs, new_tests, weight_decay, True, name="_m3")
