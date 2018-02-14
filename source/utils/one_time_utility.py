from data_manager.path_manager import PathManager
import image_loader.image_loader as il
import numpy as np
import matplotlib.image as img
import utils as utils

# Here you should write some scripts that are intended to be run once to prepare and reformat data

def reformat_train_red():
    path_man = PathManager()
    datapath = path_man.resources_path('train/train_red.mat')

    data_bad_formatted = np.matrix.transpose(il.matrix_loader(datapath))
    data = np.array([np.matrix.transpose(x) for x in data_bad_formatted]) / 255.0

    flags = il.matrix_loader(datapath, field_name='Y_red')

    il.save_mat(path_man.resources_path('train/train_red_reformatted.mat'), X=data, Y=flags)


def make_sprite(data_path='train/train_red_reformatted.mat', img_path='train/svhn_marked.png'):
    path_man = PathManager()
    datapath = path_man.resources_path(data_path)

    digit_dict = {10: (.0, .0, .0),
                  1: (1.0, .0, .0),
                  2: (.0, 1.0, .0),
                  3: (.0, .0, 1.0),
                  4: (.0, 1.0, 1.0),
                  5: (1.0, 1.0, .0),
                  6: (1.0, 1.0, 1.0),
                  7: (1.0, .0, 1.0),
                  8: (.5, .5, .5),
                  9: (1.0, .5, 1.0)}
    data = il.load(datapath, field_name='^X.*$', force_format=[32, 32, 3])[:, :, :, :]
    labels = il.matrix_loader(datapath, field_name='^Y.*$')[:, :]
    sprite = utils.make_sprite(utils.mark_images(data, labels, label_to_rgb_dict=digit_dict), force_square=True)

    img.imsave(path_man.resources_path(img_path), sprite)


make_sprite()
