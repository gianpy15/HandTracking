from hands_bounding_utils.hands_locator_from_rgbd import read_dataset_random
import numpy as np
import keras.models as km
from neural_network.keras.custom_layers.heatmap_loss import my_loss
from skimage.transform import resize
import os
from data_manager.path_manager import PathManager
import hands_bounding_utils.utils as u

pm = PathManager()

dataset_path = pm.resources_path(os.path.join("samples_for_heatmaps"))
model_ck_path = pm.resources_path(os.path.join('models/hand_cropper/cropper_v3.ckp'))
model_save_path = pm.resources_path(os.path.join('models/hand_cropper/cropper_v3.h5'))

images = read_dataset_random(path=dataset_path, number=10)[0]
np.random.shuffle(images)

images = np.array(images)
images = images / 255


def rgb2gray(rgb):
    gray = np.dot(rgb[:, :, :3], [0.2989, 0.5870, 0.1140])
    return np.reshape(gray, newshape=np.shape(gray) + (1,))


def gray2rgb(gray):
    r = (1/0.2989) * gray
    g = (1/0.5870) * gray
    b = (1/0.1140) * gray
    return np.concatenate((r, g, b), axis=-1)


# Build up the model
model = km.load_model(model_save_path, custom_objects={'my_loss': my_loss})
model.load_weights(model_ck_path)
model.summary()

# Testing the model getting some outputs
net_out = model.predict(images[0:2])[0]
net_out = net_out.clip(max=1)
first_out = resize(net_out, output_shape=(120, 160, 1))
total_sum = np.sum(first_out[0])

first_image = rgb2gray(images[0])

u.showimage(images[0])
u.showimage(u.heatmap_to_rgb(net_out))
u.showimages(u.get_crops_from_heatmap(images[0], np.squeeze(net_out), 4, 4, enlarge=1,
                                      accept_crop_minimum_dimension_pixels=100))
