import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..")))

from hands_bounding_utils.egohand_dataset_manager import create_dataset as egocreate
from hands_bounding_utils.hands_locator_from_rgbd import create_dataset_shaded_heatmaps as cropscreate
from hands_regularizer import regularizer
from junctions_locator_utils.junction_locator_ds_management import create_dataset as jointscreate
from data_manager.path_manager import *


def build_default_egohands():
    egocreate(savepath=crops_path(),
              resize_rate=0.5,
              width_shrink_rate=4,
              heigth_shrink_rate=4)


def build_default_crops():
    cropscreate(savepath=crops_path(), fillgaps=True,
                resize_rate=0.5,
                width_shrink_rate=4,
                heigth_shrink_rate=4)


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
    # build_default_crops()
