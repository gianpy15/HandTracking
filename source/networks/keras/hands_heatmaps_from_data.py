from hands_bounding_utils.hands_locator_from_rgbd import read_dataset, create_dataset
import hands_bounding_utils.utils as u
from networks.loss_function.heatmap_loss import heatmap_loss
import numpy as np
import keras.models as km
import keras.layers as kl
import keras.callbacks as kc
import keras.optimizers as ko
import os
from data_manager.path_manager import PathManager
from tensorboard.tensorboard_manager import TensorBoardManager as TBManager
import tensorflow as tf

pm = PathManager()

dataset_path = pm.resources_path(os.path.join("hands_bounding_dataset", "network_test"))
tensorboard_path = pm.resources_path(os.path.join("tbdata/heat_maps"))

TBManager.set_path("heat_maps")
tb_manager = TBManager()

# create_dataset(["handsMichele"], savepath=dataset_path, fillgaps=True,
#                resize_rate=0.25, width_shrink_rate=4, heigth_shrink_rate=4)
images, heat_maps, depths = read_dataset(path=dataset_path)

images, heat_maps = np.array(images), np.array(heat_maps)
images, heat_maps = images / 255, heat_maps / 255
heat_maps = np.reshape(heat_maps, newshape=np.shape(heat_maps) + (1,))
train_imgs, test_imgs = images[:np.shape(images)[0]//2], images[np.shape(images)[0]//2:]
train_maps, test_maps = heat_maps[:np.shape(images)[0]//2], heat_maps[np.shape(images)[0]//2:]

train_imgs, train_maps = train_imgs[0:20], train_maps[0:20]
test_imgs, test_maps = test_imgs[0:10], test_maps[0:10]

print(np.shape(train_imgs), np.shape(train_maps))

tb_manager.add_images(train_imgs, name="train_imgs")
tb_manager.add_images(train_maps, name="train_maps")

# Build up the model
model = km.Sequential()
model.add(kl.Conv2D(input_shape=np.shape(images)[1:], filters=16, kernel_size=[3, 3], padding='same'))
model.add(kl.Activation('relu'))
model.add(kl.Conv2D(filters=32, kernel_size=[3, 3], padding='same', activation='relu'))
model.add(kl.MaxPooling2D())
model.add(kl.Conv2D(filters=32, kernel_size=[3, 3], padding='same', activation='relu'))
model.add(kl.MaxPooling2D())
model.add(kl.Conv2D(filters=16, kernel_size=[3, 3], padding='same', activation='relu'))
model.add(kl.Conv2D(filters=1, kernel_size=[3, 3], padding='same'))
model.add(kl.Activation('sigmoid'))
model.summary()

# Callbacks for keras
tensor_board = kc.TensorBoard(log_dir=tensorboard_path, histogram_freq=1)
# model_ckp = kc.ModelCheckpoint(filepath=model_path, monitor=['accuracy'],
#                                verbose=1, save_best_only=True, mode='max', period=1)
es = kc.EarlyStopping(patience=20, verbose=1, monitor=['val_acc'], mode='max')
callbacks = [tensor_board]

optimizer = ko.adam(lr=1e-3)
loss = heatmap_loss
model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

print('training starting...')
model.fit(train_imgs, train_maps, epochs=30,  batch_size=5, callbacks=callbacks, verbose=2, validation_data=(test_imgs, test_maps))
print('training complete!')

# Testing the model getting some outputs
first_out = model.predict(train_imgs[0:5], batch_size=5)
tb_manager.add_images(first_out, name='output', max_out=5)


# Writing tensorboard data
init = tf.variables_initializer([])
with tf.Session() as s:
    s.run(init)
    # tb_manager.clear_data()
    summary = s.run(tb_manager.get_runnable())[0]
    tb_manager.write_step(summary, 2)
