from hands_bounding_utils.hands_locator_from_rgbd import *
from neural_network.keras.custom_layers import heatmap_loss
from neural_network.keras.models.heatmap import *
from neural_network.keras.callbacks.image_writer import ImageWriter
import os
from data_manager.path_manager import PathManager
from tensorboard.tensorboard_manager import TensorBoardManager as TBManager

pm = PathManager()

dataset_path = pm.resources_path(os.path.join("hands_bounding_dataset", "network_test"))
tensorboard_path = pm.resources_path(os.path.join("tbdata/heat_maps"))
model_ck_path = pm.resources_path(os.path.join('models/hand_cropper/cropper_v3.ckp'))
model_save_path = pm.resources_path(os.path.join('models/hand_cropper/cropper_v3.h5'))

TBManager.set_path("heat_maps")
tb_manager = TBManager('images')
train = True
random_dataset = True
shuffle = True
build_dataset = False
attach_depth = False

# Hyper parameters
train_samples = 1000
test_samples = 100
weight_decay = kr.l2(1e-5)
learning_rate = 1e-3

# Data set stuff

if build_dataset:
    create_dataset(savepath=dataset_path, fillgaps=True,
                   resize_rate=0.25, width_shrink_rate=4, heigth_shrink_rate=4)

if random_dataset:
    images, heat_maps, depths = read_dataset_random(path=dataset_path, number=train_samples + test_samples + 10)
else:
    images, heat_maps, depths = read_dataset(path=dataset_path)

if shuffle:
    images, depths, heat_maps = shuffle_rgb_depth_heatmap(images, depths, heat_maps)

images, heat_maps, depths = np.array(images), np.array(heat_maps), np.array(depths)
images = images / 255
heat_maps = np.reshape(heat_maps, newshape=np.shape(heat_maps) + (1,))
depths = np.reshape(depths, newshape=np.shape(depths) + (1,))
train_imgs, test_imgs = images[:np.shape(images)[0]//2], images[np.shape(images)[0]//2:]
train_maps, test_maps = heat_maps[:np.shape(images)[0]//2], heat_maps[np.shape(images)[0]//2:]
train_depths, test_depths = depths[:np.shape(depths)[0]//2], depths[np.shape(depths)[0]//2:]

train_imgs, train_maps, train_depths = train_imgs[0:train_samples], train_maps[0:train_samples], train_depths[0:train_samples]
test_imgs, test_maps, test_depths = test_imgs[0:test_samples], test_maps[0:test_samples], test_depths[0:test_samples]

print("Train depths = {}".format(np.shape(train_depths)))
print("Train images = {}, train maps = {}".format(np.shape(train_imgs), np.shape(train_maps)))

print("Test images = {}, test maps = {}".format(np.shape(test_imgs), np.shape(test_depths)))

if attach_depth:
    X = np.concatenate((train_imgs, train_depths), axis=-1)
    X_test = np.concatenate((test_imgs, test_depths), axis=-1)
    print("Input shape: {}".format(np.shape(X)))

model_input = X if attach_depth else train_imgs
model_test = X_test if attach_depth else test_imgs

tb_manager.add_images(test_imgs[0:5], name="test_imgs", max_out=5)
tb_manager.add_images(test_maps[0:5], name="test_maps", max_out=5)
tb_manager.add_images(train_imgs[0:5], name="train_imgs", max_out=5)
tb_manager.add_images(train_maps[0:5], name="train_maps", max_out=5)


# Build up the model
model = high_fov_model(input_shape=np.shape(model_input)[1:], weight_decay=weight_decay)
model.summary()

# Callbacks for keras
tensor_board = kc.TensorBoard(log_dir=tensorboard_path, histogram_freq=1)
model_ckp = kc.ModelCheckpoint(filepath=model_ck_path, monitor='val_loss',
                               verbose=1, save_best_only=True, mode='min', period=1)
es = kc.EarlyStopping(patience=10, verbose=1, monitor='val_loss', mode='min', min_delta=2e-4)
im = ImageWriter(images=train_imgs[0:5], tb_manager=tb_manager, name='train_output')
im2 = ImageWriter(images=test_imgs[0:5], tb_manager=tb_manager, name='test_output')
callbacks = [tensor_board, model_ckp, es, im, im2]

# Training tools
optimizer = ko.adam(lr=learning_rate)


def my_loss(heat_ground, heat_pred):
    return heatmap_loss.prop_heatmap_loss(heat_ground, heat_pred, white_priority=-1.5)


loss = my_loss
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# tb_manager.clear_data()

if train:
    print('training starting...')
    model.fit(model_input, train_maps, epochs=100, batch_size=20, callbacks=callbacks, verbose=0,
              validation_data=(model_test, test_maps))
    print('training complete!')

    model.save(model_save_path)
    print("Model saved")
    # Testing the model getting some outputs
    first_out = model.predict(model_test[0:5])
    first_out = first_out.clip(min=0)
    total_sum = np.sum(first_out[0])
    print("Total output sum = {}".format(total_sum))

    print(np.shape(first_out[0]))

    plt.imshow(np.reshape(first_out[0], newshape=(30, 40)))

    plt.show()
