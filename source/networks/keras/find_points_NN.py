from neural_network.keras.models.joints import *
from neural_network.keras.utils.model_trainer import train_model
from neural_network.keras.utils.naming import *
from neural_network.keras.utils.data_loader import load_dataset
import numpy as np

if __name__ == '__main__':

    dataset = load_dataset(train_samples=1, valid_samples=1, data_format=JLOCATOR,
                           random_dataset=True,
                           shuffle=True,
                           verbose=True,
                           separate_valid=False)

    model = train_model(model_generator=lambda: high_fov_model(weight_decay=kl.regularizers.l2(1e-5)),
                        dataset=dataset,
                        epochs=1,
                        tb_path=None,
                        patience=5,
                        verbose=True)
