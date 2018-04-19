import os
import sys

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..", "..")))

from keras.models import Model
from skimage.transform import rescale
from skimage.transform import resize
from data.datasets.data_loader import load_dataset
import data.datasets.crop.utils as u
from data.datasets.io.image_loader import load
from library.neural_network.keras.models.heatmap import *
from data.naming import *
from library.telegram.telegram_bot import *
from library.neural_network.keras.models.opposite_bias_model import opposite_bias_adversarial
from library.utils.visualization_utils import get_image_with_mask

png_path = resources_path("gui", "hands.png")
model1_save_path = cropper_h5_path('cropper_opposite_bias_v1_m1')
model2_save_path = cropper_h5_path('cropper_opposite_bias_v1_m2')
model3_save_path = cropper_h5_path('cropper_opposite_bias_v1_m3')

read_from_png = False

height = 4*50
width = 4*50
if read_from_png:
    images = load(png_path, force_format=(height, width, 3))
else:
    images = load_dataset(train_samples=2,
                          valid_samples=0,
                          data_format=CROPPER)[TRAIN_IN]

images = (images - np.mean(images))/np.std(images)

# Build up the model
model1 = km.load_model(model1_save_path)

# Testing the model getting some outputs

net_out = model1.predict(images)
k = 0.15
imgs = get_image_with_mask(images, net_out)
for idx in range(len(images)):
    min = np.min(imgs[idx])
    max = np.max(imgs[idx])
    u.showimage((imgs[idx] - min)/(max-min))
    u.showimages(u.get_crops_from_heatmap(images[idx], np.squeeze(net_out[idx]), 4, 4, enlarge=0.5,
                                          accept_crop_minimum_dimension_pixels=100))

send_to_telegram = False
if send_to_telegram:
    send_image_from_array(get_image_with_mask(images[0], net_out))
