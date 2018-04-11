import sys
import os

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "../..")))

from neural_network.keras.models.heatmap import *
from neural_network.keras.callbacks.image_writer import ImageWriter
from neural_network.keras.custom_layers.heatmap_loss import my_loss
from data_manager.path_manager import resources_path
from tensorboard_utils.tensorboard_manager import TensorBoardManager as TBManager
from neural_network.keras.utils.data_loader import *

dataset_path = resources_path(os.path.join("hands_bounding_dataset", "network_test"))
tensorboard_path = resources_path(os.path.join("tbdata/heat_maps"))
model_ck_path = resources_path(os.path.join('models/hand_cropper/cropper_v5.ckp'))
model_save_path = resources_path(os.path.join('models/hand_cropper/cropper_v5.h5'))

TBManager.set_path("heat_maps")
tb_manager_train = TBManager()
tb_manager_test = TBManager()
train = False

# Hyper parameters
weight_decay = kr.l2(1e-5)
learning_rate = 1e-3

dataset = load_dataset(train_samples=4000,
                       valid_samples=1000,
                       dataset_path=dataset_path,
                       random_dataset=True,
                       shuffle=True,
                       use_depth=False,
                       verbose=True)
tb_manager_test.add_images(dataset[VALID_IN][0:5], name="test_imgs", max_out=5)
tb_manager_test.add_images(dataset[VALID_TARGET][0:5], name="test_maps", max_out=5)
tb_manager_train.add_images(dataset[TRAIN_IN][0:5], name="train_imgs", max_out=5)
tb_manager_train.add_images(dataset[TRAIN_TARGET][0:5], name="train_maps", max_out=5)

# Build up the model
model = high_fov_model(input_shape=np.shape(dataset[TRAIN_IN])[1:], weight_decay=weight_decay)
model.summary()

# Callbacks for keras
tensor_board = kc.TensorBoard(log_dir=tensorboard_path, histogram_freq=1)
model_ckp = kc.ModelCheckpoint(filepath=model_ck_path, monitor='val_loss',
                               verbose=1, save_best_only=True, mode='min', period=1)
es = kc.EarlyStopping(patience=5, verbose=1, monitor='val_loss', mode='min', min_delta=2e-4)
im = ImageWriter(images=dataset[TRAIN_IN][0:5], tb_manager=tb_manager_train, name='train_output')
im2 = ImageWriter(images=dataset[VALID_IN][0:5], tb_manager=tb_manager_test, name='test_output')
callbacks = [tensor_board, model_ckp, es, im, im2]

# Training tools
optimizer = ko.adam(lr=learning_rate)
loss = my_loss
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# tb_manager.clear_data()

if train:
    print('training starting...')
    model.fit(dataset[TRAIN_IN], dataset[TRAIN_TARGET], epochs=50, batch_size=10, callbacks=callbacks, verbose=1,
              validation_data=(dataset[VALID_IN], dataset[VALID_TARGET]))
    print('training complete!')

    model.save(model_save_path)
    print("Model saved")
    # Testing the model getting some outputs
    first_out = model.predict(dataset[VALID_IN][0:5])
    first_out = first_out.clip(min=0)
    total_sum = np.sum(first_out[0])
    print("Total output sum = {}".format(total_sum))

    print(np.shape(first_out[0]))

    plt.imshow(np.reshape(first_out[0], newshape=(30, 40)))

    plt.show()
