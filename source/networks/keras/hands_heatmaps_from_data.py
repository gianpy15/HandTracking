import sys
import os

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "../..")))

from hands_bounding_utils.hands_locator_from_rgbd import *
from neural_network.keras.models.heatmap import *
from neural_network.keras.callbacks.image_writer import ImageWriter
from neural_network.keras.custom_layers.heatmap_loss import my_loss
from data_manager.path_manager import resources_path
from tensorboard_utils.tensorboard_manager import TensorBoardManager as TBManager
from hands_regularizer.regularizer import Regularizer
import random as rnd

dataset_path = resources_path(os.path.join("hands_bounding_dataset", "network_test"))
tensorboard_path = resources_path(os.path.join("tbdata/heat_maps"))
model_ck_path = resources_path(os.path.join('models/hand_cropper/cropper_v4.ckp'))
model_save_path = resources_path(os.path.join('models/hand_cropper/cropper_v4.h5'))

TBManager.set_path("heat_maps")
tb_manager_train = TBManager()
tb_manager_test = TBManager()
train = False
random_dataset = True
shuffle = True
build_dataset = False
attach_depth = False

# Hyper parameters
train_samples = 4000
test_samples = 200
weight_decay = kr.l2(1e-5)
learning_rate = 1e-3

validation_set_proportion = 0.3

reg = Regularizer()
reg.normalize()

# Data set stuff

basedir = pm.resources_path(os.path.join("framedata"))
vids = [x for x in os.listdir(basedir) if x not in ['.DS_Store',
                                                    'compress.sh',
                                                    'expand.sh',
                                                    'contributors.txt']]
rnd.shuffle(vids)

split_idx = int(validation_set_proportion * len(vids))
test_vids = vids[0:split_idx]
train_vids = vids[split_idx + 1:]
print("test videos: " + str(test_vids))
print("train videos: " + str(train_vids))

if build_dataset:
    create_dataset(savepath=dataset_path, fillgaps=True,
                   resize_rate=0.5, width_shrink_rate=4, heigth_shrink_rate=4)

if random_dataset:
    train_imgs, train_maps, train_depths = read_dataset_random(path=dataset_path,
                                                               number=train_samples,
                                                               leave_out=test_vids)
    test_imgs, test_maps, test_depths = read_dataset_random(path=dataset_path,
                                                            number=test_samples,
                                                            leave_out=train_vids)

else:
    train_imgs, train_maps, train_depths, test_imgs, test_maps, test_depths = read_dataset(path=dataset_path,
                                                                                           leave_out=test_vids)

if shuffle:
    train_imgs, train_depths, train_maps = shuffle_rgb_depth_heatmap(train_imgs, train_depths, train_maps)

train_imgs = np.array(train_imgs)
test_imgs = np.array(test_imgs)
train_maps = np.array(train_maps)
test_maps = np.array(test_maps)
train_depths = np.array(train_depths)
test_depths = np.array(test_depths)

train_imgs = reg.apply_on_batch(train_imgs)
test_imgs = reg.apply_on_batch(test_imgs)

train_maps = np.reshape(train_maps, newshape=np.shape(train_maps) + (1,))
train_depths = np.reshape(train_depths, newshape=np.shape(train_depths) + (1,))
test_maps = np.reshape(test_maps, newshape=np.shape(test_maps) + (1,))
test_depths = np.reshape(test_depths, newshape=np.shape(test_depths) + (1,))

train_imgs, train_maps, train_depths = train_imgs[0:train_samples], train_maps[0:train_samples], train_depths[
                                                                                                 0:train_samples]
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

tb_manager_test.add_images(test_imgs[0:5], name="test_imgs", max_out=5)
tb_manager_test.add_images(test_maps[0:5], name="test_maps", max_out=5)
tb_manager_train.add_images(train_imgs[0:5], name="train_imgs", max_out=5)
tb_manager_train.add_images(train_maps[0:5], name="train_maps", max_out=5)

# Build up the model
model = high_fov_model(input_shape=np.shape(model_input)[1:], weight_decay=weight_decay)
model.summary()

# Callbacks for keras
tensor_board = kc.TensorBoard(log_dir=tensorboard_path, histogram_freq=1)
model_ckp = kc.ModelCheckpoint(filepath=model_ck_path, monitor='val_loss',
                               verbose=1, save_best_only=True, mode='min', period=1)
es = kc.EarlyStopping(patience=5, verbose=1, monitor='val_loss', mode='min', min_delta=2e-4)
im = ImageWriter(images=train_imgs[0:5], tb_manager=tb_manager_train, name='train_output')
im2 = ImageWriter(images=test_imgs[0:5], tb_manager=tb_manager_test, name='test_output')
callbacks = [tensor_board, model_ckp, es, im, im2]

# Training tools
optimizer = ko.adam(lr=learning_rate)
loss = my_loss
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# tb_manager.clear_data()

if train:
    print('training starting...')
    model.fit(model_input, train_maps, epochs=50, batch_size=100, callbacks=callbacks, verbose=1,
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
