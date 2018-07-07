from library.hand_vector_field.finger_field import FingerField
from library.geometry.formatting import *
import numpy as np
from data.datasets.jlocator.heatmaps_to_hand import heatmaps_to_hand


def build_hand_fields(joints_data):
    print(joints_data.shape)
    hand = heatmaps_to_hand(joints=joints_data,
                            visibility=np.zeros(shape=(21,)))
    ff = {}
    for finger in FINGERS:
        ff[finger] = FingerField(finger=hand[finger],
                                 img_dims=joints_data.shape)

    return np.concatenate([ff[finger].field for finger in FINGERS], axis=2)


if __name__ == '__main__':
    from matplotlib import pyplot as mplt
    from data import *
    from library.utils.visualization_utils import joint_skeleton_impression
    from time import time
    from skimage.transform import resize

    dm = DatasetManager(train_samples=1,
                        valid_samples=1,
                        batch_size=1,
                        dataset_dir=joints_path(),
                        formatting=JUNC_STD_FORMAT)
    data = dm.train()
    plain = data[0][IN(0)][0]
    t1 = time()
    fields = build_hand_fields(data[0][OUT(0)][0])
    t3 = time()
    print((t3-t1))
    f_repr = np.sum([fields[:, :, 2*i:2*i+2] for i in range(fields.shape[-1]//2)], axis=0)
    f_repr = np.concatenate([f_repr, np.zeros(shape=f_repr.shape[:-1]+(1,))], axis=2)
    # f_repr = (f_repr + 1.0)/2

    highlighted = plain+resize(f_repr, output_shape=plain.shape)
    highlighted = highlighted / np.max(highlighted)

    mplt.imshow(highlighted)
    mplt.show()
    mplt.imshow(plain)
    mplt.show()
    skeletal = joint_skeleton_impression(feed=data[0],
                                         heats_key=OUT(0),
                                         vis_key=OUT(1))
    print(np.shape(skeletal))
    mplt.imshow(skeletal[0])
    mplt.show()