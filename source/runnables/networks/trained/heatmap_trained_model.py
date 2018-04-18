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
from library.utils.visualization_utils import get_image_with_mask

dataset_path = resources_path("hands_bounding_dataset", "network_test")
png_path = resources_path("gui", "hands.png")
model1_save_path = models_path('hand_cropper', 'incremental_predictor', 'cropper_v5_m1.h5')
model2_save_path = models_path('hand_cropper', 'incremental_predictor', 'cropper_v5_m2.h5')
model3_save_path = models_path('hand_cropper', 'incremental_predictor', 'cropper_v5_m3.h5')

read_from_png = False

height = 4*50
width = 4*50
if read_from_png:
    images = load(png_path, force_format=(height, width, 3))
else:
    images = load_dataset(train_samples=2,
                          valid_samples=0,
                          dataset_path=dataset_path)[TRAIN_IN]

images_ = (images - np.mean(images))/np.std(images)
# images_ = images
images_ = np.concatenate((images_, np.zeros(shape=np.shape(images_)[0:-1] + (1,))), axis=-1)
print(np.shape(images_))


def attach_heat_map(inputs, fitted_model: Model):
    _inputs = inputs[:, :, :, 0:3]
    outputs = fitted_model.predict(inputs)
    rescaled = []
    for img in outputs:
        rescaled.append(rescale(img, 4.0))
    outputs = np.array(rescaled)
    inputs_ = np.concatenate((_inputs, outputs), axis=-1)
    return inputs_


# Build up the model
model1 = km.load_model(model1_save_path)

images_ = attach_heat_map(images_, model1)
model2 = km.load_model(model2_save_path)

images_ = attach_heat_map(images_, model2)
model = km.load_model(model3_save_path)

# Testing the model getting some outputs
net_out = model.predict(images_)
net_out = net_out.clip(max=1)
images = (images - np.mean(images))/np.var(images)
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
