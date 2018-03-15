from hands_bounding_utils.hands_locator_from_rgbd import read_dataset, create_dataset
from networks.custom_layers.heatmap_loss import heatmap_loss
import numpy as np
import keras.models as km
import keras.layers as kl
import keras.callbacks as kc
import keras.optimizers as ko
import matplotlib.pyplot as plt
import keras.losses as klo
from networks.custom_layers.abs import Abs
import os
from data_manager.path_manager import PathManager
from tensorboard.tensorboard_manager import TensorBoardManager as TBManager
import tensorflow as tf

pm = PathManager()

dataset_path = pm.resources_path(os.path.join("hands_bounding_dataset", "network_test"))
tensorboard_path = pm.resources_path(os.path.join("tbdata/heat_maps"))
model_ck_path = pm.resources_path(os.path.join('models/hand_cropper/cropper_v1.ckp'))
model_save_path = pm.resources_path(os.path.join('models/hand_cropper/cropper_v1.h5'))

TBManager.set_path("heat_maps")
tb_manager = TBManager('images')
train = True
train_samples = 100
test_samples = 20

# create_dataset(["handsMichele"], savepath=dataset_path, fillgaps=True,
#                resize_rate=0.25, width_shrink_rate=4, heigth_shrink_rate=4)
images, heat_maps, depths = read_dataset(path=dataset_path)

images, heat_maps, depths = np.array(images), np.array(heat_maps), np.array(depths)
images, depths = images / 255, depths / 255
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

X = np.concatenate((train_imgs, train_depths), axis=-1)
X_test = np.concatenate((test_imgs, test_depths), axis=-1)
print("Input shape: {}".format(np.shape(X)))

model_input = train_imgs
model_test = test_imgs

tb_manager.add_images(test_imgs, name="train_imgs", max_out=5)
tb_manager.add_images(test_maps, name="train_maps", max_out=5)

# Build up the model
model = km.Sequential()
model.add(kl.Conv2D(input_shape=np.shape(model_input)[1:], filters=32, kernel_size=[3, 3], padding='same'))
model.add(kl.Activation('relu'))
model.add(kl.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation='relu'))
model.add(kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation='relu'))
model.add(kl.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation='relu'))
model.add(kl.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation='relu'))
model.add(kl.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation='relu'))
model.add(kl.MaxPooling2D())
model.add(kl.Conv2D(filters=16, kernel_size=[3, 3], padding='same', activation='relu'))
model.add(kl.MaxPooling2D())
model.add(kl.Conv2D(filters=1, kernel_size=[3, 3], padding='same'))
# model.add(Softmax4D(axis=1, name='softmax4D'))
model.add(Abs())
model.summary()

# Callbacks for keras
tensor_board = kc.TensorBoard(log_dir=tensorboard_path, histogram_freq=1)
model_ckp = kc.ModelCheckpoint(filepath=model_ck_path, monitor='val_loss',
                               verbose=1, save_best_only=True, mode='min', period=1)
es = kc.EarlyStopping(patience=10, verbose=1, monitor='val_loss', mode='min')
callbacks = [tensor_board, model_ckp, es]

optimizer = ko.adam(lr=1e-3)
loss = heatmap_loss
model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

# tb_manager.clear_data()

if train:
    print('training starting...')
    model.fit(model_input, train_maps, epochs=100,  batch_size=20, callbacks=callbacks, verbose=1, validation_data=(model_test, test_maps))
    print('training complete!')

    model.save(model_save_path)
    print("Model saved")
    # Testing the model getting some outputs
    first_out = model.predict(model_test[0:5])
    first_out = first_out.clip(min=0)
    tb_manager.add_images(first_out, name='output', max_out=5)
    total_sum = np.sum(first_out[0])
    print("Total output sum = {}".format(total_sum))

    # Writing tensorboard data
    init = tf.variables_initializer([])
    print("starting session..")
    with tf.Session() as s:
        print("I'm inside")
        s.run(init)
        # tb_manager.clear_data()
        summary = s.run(tb_manager.get_runnable())[0]
        tb_manager.write_step(summary, 40)

    print('session finished')

    print(np.shape(first_out[0]))

    plt.imshow(np.reshape(first_out[0], newshape=(30, 40)))

    plt.show()
