import source.hands_bounding_utils.hands_dataset_manager as data
import source.hands_bounding_utils.egohand_dataset_manager as ego
from data_manager import path_manager as pm
from source.utils import utils
import keras.models as km
import keras.layers as kl
import numpy as np
import keras.callbacks as kc


# ############# PATHS ############
train_images_path = ego.default_train_images_path()
train_annots_path = ego.default_train_annotations_path()
# test_images_path = ego.default_test_images_path()
# test_annots_path = ego.default_test_annotations_path()
tb_dir = '../../../resources/tensorboard/tbdata'
model_path = '../../../resources/models/hand_cropper/cropper_v1.ckp'


# ################ BASIC PARAMETERS #####################
train_set_dim = 1
test_set_dim = 1
batch_dimension = 1
learning_rate = 0.005
epochs = 40
height_shrink_rate = 10
width_shrink_rate = 10

# ################## GETTING DATA SETS #######################
train_images, train_heatmaps = ego.get_random_samples_from_dataset(train_images_path,
                                                                    train_annots_path, train_set_dim + test_set_dim,
                                                                    height_shrink_rate, width_shrink_rate)
test_images, test_heatmaps = train_images[-test_set_dim:], train_heatmaps[-test_set_dim:]
train_images = train_images[0: train_set_dim - test_set_dim]
train_heatmaps = train_heatmaps[0: train_set_dim - test_set_dim]

# test_images, test_heatmaps = ego.get_random_samples_from_dataset(test_images_path,
#                                                                   test_annots_path, test_set_dim,
#                                                                   height_shrink_rate, width_shrink_rate)
# for img in train_images:
#     np.transpose(img)
# for img in test_images:
#     np.transpose(img)
# for img in train_heatmaps:
#     np.transpose(img)
# for img in test_heatmaps:
#     np.transpose(img)
train_images = np.array(train_images)
# test_images = np.array(test_images)
train_heatmaps = np.array(train_heatmaps)
# test_heatmaps = np.array(test_heatmaps)
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
model.add(kl.Conv2D(input_shape=(None, None, 3), filters=filters_conv1,
                    kernel_size=kernel_dim_conv1, padding='same'))
model.add(kl.Activation(activation='relu'))
model.add(kl.Conv2D(filters=filters_conv2,
                    kernel_size=kernel_dim_conv2, padding='same'))
model.add(kl.Activation(activation='relu'))
model.add(kl.Conv2D(filters=filters_conv3,
                    kernel_size=kernel_dim_conv3, padding='same'))
model.add(kl.Activation(activation='relu'))
model.add(kl.Conv2D(filters=filters_conv4,
                    kernel_size=kernel_dim_conv4, padding='same'))
model.add(kl.Activation(activation='relu'))
model.add(kl.Conv2D(filters=filters_conv5,
                    kernel_size=kernel_dim_conv5, padding='same'))
model.add(kl.Activation(activation='relu'))
model.add(kl.MaxPool2D(pool_size=pooling_size))
model.add(kl.Activation('sigmoid'))

tensor_board = kc.TensorBoard(log_dir=tb_dir, histogram_freq=20, write_grads=1, write_images=1)
model_ckp = kc.ModelCheckpoint(filepath=model_path, monitor=['accuracy'],
                               verbose=1, save_best_only=True, mode='max', period=1)
es = kc.EarlyStopping(patience=20, verbose=1, monitor=['val_acc'], mode='max')

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.summary()

print('training starting...')
history = model.fit(train_images, train_heatmaps, epochs=50,  batch_size=5,
                    callbacks=[tensor_board, model_ckp, es], verbose=2, validation_data=(test_images, test_heatmaps))
print('training complete!')

