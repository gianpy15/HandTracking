import sys
import os

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..")))

from neural_network.keras.models.heatmap import high_fov_model
from neural_network.keras.utils.data_loader import load_dataset
from neural_network.keras.utils.naming import *
from neural_network.keras.utils.model_trainer import train_model
import keras.regularizers as kr
from neural_network.keras.utils.data_augmenter import Augmenter
from hands_regularizer.regularizer import Regularizer

tensorboard_path = tensorboard_path("heat_maps")

train = True

# Load data
dataset = load_dataset(train_samples=4000,
                       valid_samples=1000,
                       random_dataset=True,
                       shuffle=True,
                       use_depth=False,
                       verbose=True)

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
                        epochs=50,
                        batch_size=40,
                        patience=5,
                        verbose=True)
