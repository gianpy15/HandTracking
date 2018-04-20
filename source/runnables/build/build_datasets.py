import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..")))

from data.datasets.crop.egohand_dataset_manager import create_dataset as egocreate
from data.datasets.crop.hands_locator_from_rgbd import create_dataset_shaded_heatmaps as cropscreate
from data.regularization import regularizer
from data.datasets.jlocator.junction_locator_ds_management import create_dataset as jointscreate
from data.naming import *


# egohands by default have dimension 720x1280
def build_default_egohands(res_rate=0.5):
    egocreate(savepath=crops_path(),
              resize_rate=res_rate,
              width_shrink_rate=1,
              heigth_shrink_rate=1)


# our hands by default have dimension 480x640
def build_default_crops(res_rate=0.5):
    cropscreate(savepath=crops_path(), fillgaps=True,
                resize_rate=res_rate,
                width_shrink_rate=1,
                heigth_shrink_rate=1)


def create_joint_dataset():
    img_reg = regularizer.Regularizer()
    img_reg.fixresize(200, 200)
    hm_reg = regularizer.Regularizer()
    hm_reg.fixresize(100, 100)
    hm_reg.heatmaps_threshold(.5)
    jointscreate(savepath=joints_path(), fillgaps=True,
                 im_regularizer=img_reg,
                 heat_regularizer=hm_reg, enlarge=.5, cross_radius=5)


if __name__ == '__main__':
    build_default_egohands()
    build_default_crops()
