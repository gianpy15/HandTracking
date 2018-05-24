import sys
import os

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..", "..")))

from library.neural_network.keras.models.palm_back_classifier import *
from library.neural_network.keras.training.model_trainer import train_model
import keras.regularizers as kr
import data.regularization.regularizer as reg
from library.neural_network.batch_processing.processing_plan import ProcessingPlan
from data import *
from data.datasets.palm_back_classifier.pb_classifier_ds_management import *
from data.naming import *

# palm visible (+1.0)
# back visible (0.0)

# ########## HYPERPARAMETERS ################

# MODEL NAME
name = "palm_back_simple_sequential"
# NUMBER OF SAMPLES IN TRAIN SET
train_samples = 700
# NUMBER OF SAMPLES IN VALIDATION SET
valid_samples = 300
# BATCH SIZE
batch_size = 11
# NUMBER OF EPOCHS
epochs = 20
# PATIENCE
patience = 10
# WEIGHT DECAY "ALPHA" COEFFICIENT (STILL NOT USED)
weight_decay = kr.l2(1e-5)
# LEARNING RATE
learning_rate = 1e-5
# CONFIDENCE USED TO CHOOSE ELEMENTS FOR THE TRAIN/TEST SETS.
# SAMPLES (.mat FILES) ARE PRODUCED BY THE CREATE_DATASET FUNCTION AND
# EACH CUT IN EACH SAMPLE IS ASSIGNED A "CONFIDENCE" AND A LABEL (PALM=1 BACK=0).
# THE HIGHER THE CONFIDENCE IS, THE BETTER THE SAMPLE IS. IT MEANS THAT A PERFECT PALM WILL HAVE
# CONFIDENCE CLOSE TO 1, AS WELL AS A PERFECT BACK WILL HAVE A CONFIDENCE CLOSE TO 1. INSTEAD,
# HARDLY RECOGNIZABALE HANDS (SUCH AS A HAND THAT IS PERFECLY "LATERAL") WILL BE ASSIGNED A CONFIDENCE
# CLOSE TO 0 (AND SO WILL BE LESS RECOGNIZABLE AND CLASSIFIABLE BOTH BY HUMANS AND THE NETWORK)
# BASICALLY, THIS PARAMETER CAN BE TUNED TO DECIDE HOW GOOD THE SAMPLES THAT THE NETWORK WILL TRAIN
# ON WILL BE.
minconf = 0.999


# SET TO TRUE TO CREATE THE DATASET. SET TO FALSE IF THE DATASET IS ALREADY CREATED
createdataset = False
# PATH AT WHICH THE DATASET WILL BE SAVED/READ
path = palmback_path()

if __name__ == '__main__':
    # NO NEED TO TOUCH ANYTHING AFTER THIS, IT'S DELICATE
    regularizer = reg.Regularizer()
    regularizer.fixresize(200, 200)
    if createdataset:
        create_dataset(savepath=path, im_regularizer=regularizer)

    formatting = confidence_filtered_pb_format(minconf)
    generator = DatasetManager(train_samples=train_samples,
                               valid_samples=valid_samples,
                               batch_size=batch_size,
                               dataset_dir=path,
                               formatting=formatting)

    data_processing_plan = ProcessingPlan()

    model1 = train_model(model_generator=lambda: simple_classifier_rgb(weight_decay=weight_decay),
                         dataset_manager=generator,
                         loss='binary_crossentropy',
                         learning_rate=learning_rate,
                         patience=patience,
                         data_processing_plan=data_processing_plan,
                         tb_path='palm_back/',
                         model_name=name,
                         model_path=resources_path(os.path.join("models", "palm_back", name)),
                         epochs=epochs,
                         enable_telegram_log=True)
