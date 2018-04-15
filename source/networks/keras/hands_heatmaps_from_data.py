import sys
import os

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..")))

from neural_network.keras.models.heatmap import high_fov_model
from neural_network.keras.utils.data_loader import load_dataset
from neural_network.keras.utils.naming import *
from neural_network.keras.utils.model_trainer import train_model
import keras.regularizers as kr

tensorboard_path = tensorboard_path("heat_maps")

train = True

dataset = load_dataset(train_samples=2,
                       valid_samples=1,
                       random_dataset=True,
                       shuffle=True,
                       use_depth=False,
                       verbose=True)

model_generator = lambda: high_fov_model(channels=3,
                                         weight_decay=kr.l2(1e-5))

if train:
    model = train_model(model_generator=model_generator,
                        dataset=dataset,
                        tb_path=tensorboard_path,
                        model_name='cropper_v6',
                        model_type=CROPPER,
                        epochs=1,
                        batch_size=1,
                        patience=5,
                        verbose=True)
