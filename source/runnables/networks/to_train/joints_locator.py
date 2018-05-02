import os
import sys

sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..", "..")))

from library.neural_network.keras.custom_layers.heatmap_loss import *
from data import *
from library import *
import keras as K
from library.neural_network.keras.models.joints import low_injection_locator
from library.neural_network.keras.training.model_trainer import train_model
from library.neural_network.batch_processing.processing_plan import ProcessingPlan
from library.utils.visualization_utils import joint_skeleton_impression

train_samples = 1
valid_samples = 1
batch_size = 20

if __name__ == '__main__':
    model = 'jlocator_lowinj'
    set_verbosity(DEBUG)

    # Hyper parameters
    learning_rate = 1e-4
    white_priority = -2.
    delta = 6
    drate = 0.2

    heatmap_loss = lambda x, y: prop_heatmap_penalized_fp_loss(x, y,
                                                               white_priority=white_priority,
                                                               delta=delta)

    JUNC_LOWINJ_FORMAT = {
        IN('img'): MIDFMT_JUNC_RGB,
        OUT('mid_heats'): MIDFMT_JUNC_HEATMAP,
        OUT('vis'): MIDFMT_JUNC_VISIBILITY,
        OUT('heats'): MIDFMT_JUNC_HEATMAP
    }

    def reduce_heatmap_by_two(heat: np.ndarray):
        heatshape = np.shape(heat)
        outheat = np.zeros(shape=(int(heatshape[0] / 2), int(heatshape[1] / 2), heatshape[2]), dtype=heat.dtype)
        for row in range(len(heat)):
            for col in range(len(heat[row])):
                outheat[int(row/2), int(col/2)] += heat[row, col] / 4
        return outheat

    JUNC_LOWINJ_FORMAT = format_add_outer_func(f=reduce_heatmap_by_two,
                                               format=JUNC_LOWINJ_FORMAT,
                                               entry=OUT('heats'))

    # Load data
    dm = DatasetManager(train_samples=train_samples,
                        valid_samples=valid_samples,
                        batch_size=batch_size,
                        dataset_dir=joints_path(),
                        formatting=JUNC_LOWINJ_FORMAT)

    # Plan the processing needed before providing inputs and outputs for training and validation
    data_processing_plan = ProcessingPlan(augmenter=Augmenter().shift_hue(.2).shift_sat(.2).shift_val(.2),
                                          # regularizer=Regularizer().normalize(),
                                          keyset={IN('img')})  # Today we just need to augment one input...
    model = train_model(model_generator=lambda: low_injection_locator(input_shape=np.shape(dm.train()[0][IN('img')][0]),
                                                                      dropout_rate=drate,
                                                                      activation=lambda: K.layers.LeakyReLU(alpha=0.1)
                                                                      ),
                        dataset_manager=dm,
                        loss={OUT('heats'): heatmap_loss,
                              OUT('mid_heats'): heatmap_loss,
                              OUT('vis'): 'binary_crossentropy'},
                        learning_rate=learning_rate,
                        patience=10,
                        data_processing_plan=data_processing_plan,
                        tb_path="joints",
                        tb_plots={'target': lambda feed: joint_skeleton_impression(feed,
                                                                                   img_key=IN('img'),
                                                                                   heats_key=OUT('heats'),
                                                                                   vis_key=OUT('vis')),
                                  'output': lambda feed: joint_skeleton_impression(feed,
                                                                                   img_key=IN('img'),
                                                                                   heats_key=NET_OUT('heats'),
                                                                                   vis_key=NET_OUT('vis')),
                                  'mid_output': lambda feed: joint_skeleton_impression(feed,
                                                                                       img_key=IN('img'),
                                                                                       heats_key=NET_OUT('mid_heats'),
                                                                                       vis_key=NET_OUT('vis'))
                                  },
                        model_name=model,
                        model_path=joint_locators_path(),
                        epochs=50,
                        enable_telegram_log=False)
