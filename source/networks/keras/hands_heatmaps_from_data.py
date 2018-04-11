import sys
import os

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "../..")))

from neural_network.keras.models.heatmap import *
from tensorboard_utils.tensorboard_manager import TensorBoardManager as TBManager
from neural_network.keras.utils.data_loader import *
from neural_network.keras.utils.model_trainer import train_model

dataset_path = resources_path(os.path.join("hands_bounding_dataset", "network_test"))
tensorboard_path = resources_path(os.path.join("tbdata/heat_maps"))
model_ck_path = resources_path(os.path.join('models/hand_cropper/cropper_v5.ckp'))
model_save_path = resources_path(os.path.join('models/hand_cropper/cropper_v5.h5'))

TBManager.set_path("heat_maps")
tb_manager_train = TBManager()
tb_manager_test = TBManager()
train = True

dataset = load_dataset(train_samples=2,
                       valid_samples=1,
                       dataset_path=dataset_path,
                       random_dataset=True,
                       shuffle=True,
                       use_depth=False,
                       verbose=True)

model = high_fov_model(input_shape=np.shape(dataset[TRAIN_IN])[1:],
                       weight_decay=kr.l2(1e-5))
model.summary()

if train:
    model = train_model(model=model,
                        dataset=dataset,
                        tb_path=tensorboard_path,
                        checkpoint_path=model_ck_path,
                        h5model_path=model_save_path,
                        learning_rate=1e-3,
                        epochs=1,
                        batch_size=1,
                        patience=5)
