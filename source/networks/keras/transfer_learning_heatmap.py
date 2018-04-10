from data_manager.path_manager import PathManager
from keras.applications.vgg16 import VGG16
from hands_bounding_utils.hands_locator_from_rgbd import *
from neural_network.keras.custom_layers import heatmap_loss
from neural_network.keras.models.heatmap import *
from neural_network.keras.callbacks.image_writer import ImageWriter
from keras.models import Model
import os
from data_manager.path_manager import PathManager
from tensorboard_utils.tensorboard_manager import TensorBoardManager as TBManager

pm = PathManager()

dataset_path = pm.resources_path(os.path.join("hands_bounding_dataset", "network_test"))
tensorboard_path = pm.resources_path(os.path.join("tbdata/heat_maps"))
model_ck_path = pm.resources_path(os.path.join('models/hand_cropper/cropper_v4.ckp'))
model_save_path = pm.resources_path(os.path.join('models/hand_cropper/cropper_v4.h5'))

TBManager.set_path("heat_maps")
tb_manager = TBManager('images')
train = True
random_dataset = True
shuffle = True
build_dataset = False
attach_depth = False

# Hyper parameters
train_samples = 100
test_samples = 10
weight_decay = kr.l2(1e-5)
learning_rate = 1e-3

# ############### Data set stuff ###############
if build_dataset:
    create_dataset(savepath=dataset_path, fillgaps=True,
                   resize_rate=1.0, width_shrink_rate=16, heigth_shrink_rate=16)

if random_dataset:
    images, heat_maps, depths = read_dataset_random(path=dataset_path, number=train_samples + test_samples)
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

tb_manager.add_images(test_imgs[0:5], name="train_imgs", max_out=5)
tb_manager.add_images(test_maps[0:5], name="train_maps", max_out=5)


# ############### model ###############

base_model = VGG16(include_top=False, weights='imagenet', input_shape=np.shape(model_input)[1:])
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
layers = base_model.layers
layers.pop()
model = km.Sequential(layers=layers)
for i in range(len(model.layers)):
    if i <= 14:
        model.layers[i].trainable = False

model.add(kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation='relu'))
model.add(kl.Conv2D(filters=1, kernel_size=[3, 3], padding='same', activation='relu'))
model.summary()


# ############### Callbacks for keras ###############
tensor_board = kc.TensorBoard(log_dir=tensorboard_path, histogram_freq=1)
model_ckp = kc.ModelCheckpoint(filepath=model_ck_path, monitor='val_loss',
                               verbose=1, save_best_only=True, mode='min', period=1)
es = kc.EarlyStopping(patience=10, verbose=1, monitor='val_loss', mode='min', min_delta=2e-4)
im = ImageWriter(images=test_imgs[0:5], tb_manager=tb_manager)
callbacks = [tensor_board, model_ckp, es, im]

# ############### Training tools ###############
optimizer = ko.adam(lr=learning_rate)
loss = heatmap_loss
model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

if train:
    print('training starting...')
    model.fit(model_input, train_maps, epochs=20, batch_size=20, callbacks=callbacks, verbose=0,
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