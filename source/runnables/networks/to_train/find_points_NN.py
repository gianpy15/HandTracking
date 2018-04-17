import sys
import os

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..", "..")))

from library.neural_network.keras.models.joints import high_fov_model
from library.neural_network.keras.training.model_trainer import train_model
from data.datasets.data_loader import load_joint_dataset
import keras as K

if __name__ == '__main__':
    dataset = load_joint_dataset(train_samples=1, valid_samples=1,
                                 random_dataset=True,
                                 shuffle=True,
                                 verbose=True,
                                 separate_valid=True)

    model = train_model(model_generator=lambda: high_fov_model(weight_decay=K.regularizers.l2(1e-5)),
                        dataset=dataset,
                        epochs=1,
                        tb_path=None,
                        patience=5,
                        verbose=True)
