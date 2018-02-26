import source.hands_bounding_utils.hands_dataset_manager as data
from data_manager import path_manager
from source.utils import utils
import keras.models as km
import keras.layers as kl
import numpy as np
import keras.callbacks as kc

pm = path_manager.PathManager()


# ############# PATHS ############
train_images_path = data.default_train_images_path()
train_annots_path = data.default_train_annotations_path()
test_images_path = data.default_test_images_path()
test_annots_path = data.default_test_annotations_path()
tb_dir = '../../../resources/tensorboard/tbdata'


# ################ BASIC PARAMETERS #####################
train_set_dim = 100
test_set_dim = 100
batch_dimension = 1
learning_rate = 0.005
epochs = 40
height_shrink_rate = 10
width_shrink_rate = 10

# ################## GETTING DATA SETS #######################
train_images, train_heatmaps = data.get_random_samples_from_dataset(train_images_path,
                                                                    train_annots_path, train_set_dim,
                                                                    height_shrink_rate, width_shrink_rate)
test_images, test_heatmaps = data.get_random_samples_from_dataset(test_images_path,
                                                                  test_annots_path, train_set_dim,
                                                                  height_shrink_rate, width_shrink_rate)
for img in train_images:
    np.transpose(img)
for img in test_images:
    np.transpose(img)
for img in train_heatmaps:
    np.transpose(img)
for img in test_heatmaps:
    np.transpose(img)
train_images = np.array(train_images)
test_images = np.array(test_images)
train_heatmaps = np.array(train_heatmaps)
test_heatmaps = np.array(test_heatmaps)
print(np.shape(train_images))

# ################## NETWORK PARAMETERS ######################
filters_conv1 = 32
kernel_dim_conv1 = 5

filters_conv2 = 64
kernel_dim_conv2 = 3

filters_conv3 = 64
kernel_dim_conv3 = 3

filters_conv4 = 32
kernel_dim_conv4 = 3

filters_conv5 = 1
kernel_dim_conv5 = 3


pooling_size = [height_shrink_rate, width_shrink_rate]

# ################## NETWORK DEFINITION ######################

model = km.Sequential()
model.add(kl.Conv2D(input_shape=(3, None, None), filters=filters_conv1,
                    kernel_size=kernel_dim_conv1, padding='same', data_format='channels_first'))
model.add(kl.Activation(activation='relu'))
model.add(kl.Conv2D(filters=filters_conv2,
                    kernel_size=kernel_dim_conv2, padding='same', data_format='channels_first'))
model.add(kl.Activation(activation='relu'))
model.add(kl.Conv2D(filters=filters_conv3,
                    kernel_size=kernel_dim_conv3, padding='same', data_format='channels_first'))
model.add(kl.Activation(activation='relu'))
model.add(kl.Conv2D(filters=filters_conv4,
                    kernel_size=kernel_dim_conv4, padding='same', data_format='channels_first'))
model.add(kl.Activation(activation='relu'))
model.add(kl.Conv2D(filters=filters_conv5,
                    kernel_size=kernel_dim_conv5, padding='same', data_format='channels_first'))
model.add(kl.Activation(activation='relu'))
model.add(kl.MaxPool2D(pool_size=pooling_size, data_format='channels_first'))
model.add(kl.Activation('sigmoid'))

# tensor_board = kc.TensorBoard(log_dir=tb_dir, histogram_freq=20, write_grads=1)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.summary()

print('training starting...')
history = model.fit(train_images, train_heatmaps, epochs=50, validation_data=(test_images, test_heatmaps),
                    callbacks=[], verbose=2)
print('training complete!')

