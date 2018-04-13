from data_manager.path_manager import *

TRAIN_IN = 'TRAIN_IN'
TRAIN_TARGET = 'TRAIN_TARGET'
VALID_IN = 'TEST_IN'
VALID_TARGET = 'TEST_TARGET'


def cropper_h5_path(name):
    return croppers_path(name+".h5")


def cropper_ckp_path(name):
    return croppers_path(name+".ckp")


def jlocator_h5_path(name):
    return joint_locators_path(name+".h5")


def jlocator_ckp_path(name):
    return joint_locators_path(name+".ckp")