from library.hand_vector_field.field_base import FieldBase
import numpy as np
from numba import jit, prange


class FingerField:
    def __init__(self, finger, img_dims: tuple):
        fbs = [FieldBase(p1=tuple(finger[idx][:2]),
                         p2=tuple(finger[idx+1][:2]),
                         img_dims=img_dims) for idx in range(len(finger)-1)]
        self.field = np.zeros_like(fbs[0].field)
        self.build_field(np.array([fb.field for fb in fbs]))

    def build_field(self, fbs: np.ndarray):
        self.field = np.sum(fbs, axis=0)
        mags = np.linalg.norm(fbs, axis=3)
        maxmags = np.max(mags, axis=0)
        mags = np.linalg.norm(self.field, axis=2)
        np.divide(maxmags, mags, out=mags, where=mags > 0)
        np.multiply(self.field, mags[:, :, None], out=self.field)


if __name__ == '__main__':
    from matplotlib import pyplot as mplt
    from data import *
    from library.geometry.formatting import *
    from library.utils.visualization_utils import joint_skeleton_impression
    from data.datasets.jlocator.heatmaps_to_hand import heatmaps_to_hand
    from time import time
    dm = DatasetManager(train_samples=1,
                        valid_samples=1,
                        batch_size=1,
                        dataset_dir=joints_path(),
                        formatting=JUNC_STD_FORMAT)
    data = dm.train()
    plain = data[0][IN(0)][0]
    t1 = time()
    hand = heatmaps_to_hand(joints=data[0][OUT(0)][0],
                            visibility=data[0][OUT(1)][0])
    t2 = time()
    ff = FingerField(finger=hand[THUMB],
                     img_dims=plain.shape)
    t3 = time()
    print((t3-t1, t3-t2, t2-t1))
    f_repr = np.concatenate((ff.field, np.zeros(shape=ff.field.shape[:-1]+(1,))), axis=2)
    # f_repr = (f_repr + 1.0)/2

    highlighted = plain+f_repr
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