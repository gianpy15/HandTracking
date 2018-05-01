import sys
import os

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..", "..")))

from library.neural_network.keras.models.heatmap import high_fov_model
from data.datasets.data_loader import load_crop_dataset
from data.naming import *
from library.neural_network.keras.training.model_trainer import train_model
import keras.regularizers as kr
from data.augmentation.data_augmenter import Augmenter
from data.regularization.regularizer import Regularizer

tensorboard_path = tensorboard_path("heat_maps")

train = True

# Load data
dataset = load_crop_dataset(train_samples=4,
                            valid_samples=1)

# Augment data
print("Augmenting data...")
augmenter = Augmenter()
augmenter.shift_hue(prob=0.25).shift_sat(prob=0.25).shift_val(prob=0.25)
dataset[TRAIN_IN] = augmenter.apply_on_batch(dataset[TRAIN_IN])
dataset[VALID_IN] = augmenter.apply_on_batch(dataset[VALID_IN])
print("Augmentation end")

# Regularize data
print("Regularizing data...")
regularizer = Regularizer()
regularizer.normalize()
dataset[TRAIN_IN] = regularizer.apply_on_batch(dataset[TRAIN_IN])
dataset[VALID_IN] = regularizer.apply_on_batch(dataset[VALID_IN])
print("Regularization end")

# train the model
model_generator = lambda: high_fov_model(channels=3,
                                         weight_decay=kr.l2(1e-5))

if train:
    model = train_model(model_generator=model_generator,
                        dataset=dataset,
                        tb_path=tensorboard_path,
                        model_name='cropper_v6',
                        model_type=CROPPER,
                        epochs=1,
                        batch_size=40,
                        patience=5,
                        enable_telegram_log=True)
