import scipy.io
import os
import numpy as np
from skimage import transform
from skimage import color
from matplotlib import image
import re


def matrix_loader(filename, field_name='X_red'):
    """
    This method loads a matrix from a file path
    :param filename: the path to the file
    :param field_name: is the name of the field on mat file
    :return: the matrix
    """
    matrix = scipy.io.loadmat(filename)
    keys = [k for k in matrix.keys() if re.compile(field_name).match(k)]
    ret = []
    for k in keys:
        ret.extend(matrix[k])
    return np.array(ret)


def save_mat(filename, **variables):
    scipy.io.savemat(filename, variables)


def matrix_printer(matrix):
    """
    This method prints the matrix passed as a parameter
    :param matrix: the matrix to be printed
    """
    print(matrix)
    return


def save_image_from_matrix(matrix, path):
    """
    This method loads an image from a 3D RGB matrix
    :param matrix: the matrix representing image pixels
    :param path: path for the png image
    :return: the image
    """
    image.imsave(path, matrix)


def load_from_png(path):
    matrix = image.imread(path)
    return 1.0 * matrix


def get_black_filter_from_png(path):
    matrix = load_from_png(path)
    matrix[matrix != 0] = 1
    return matrix


def load(path, field_name=None, force_format=None, affine_transform=None, alpha=False):
    out = []
    # if path is not a string, then assume it is a collection of paths
    if not isinstance(path, str):
        # solve recursively for its entire depth, then return
        for p in path:
            batch = load(p, field_name=field_name, force_format=force_format, affine_transform=affine_transform)
            if batch is not None:
                out.extend(batch)
        return np.array(out)

    # base step if path is an effective path
    # choose action based on extension
    ext = os.path.splitext(path)[-1]
    if ext in ['.png', '.jpg']:
        data = load_from_png(path)
    elif ext == '.mat':
        if field_name is not None:
            data = matrix_loader(path, field_name=field_name)
        else:
            data = matrix_loader(path)
    else:
        return None

    # if data is RGBA and alpha channel is unset, then cut away the alpha channel
    if not alpha and data.shape[-1] == 4:
        if len(data.shape) == 4:
            data = data[:, :, :, 0:3]
        else:
            data = data[:, :, 0:3]
    # if we have a single image in grayscale or a batch of grayscale without channel, make them well formatted
    if len(data.shape) == 2 or (len(data.shape) == 3 and data.shape[-1] > 4):
        data = np.reshape(data, data.shape+(1,))
    # we process data in batches, so format data like that
    if len(data.shape) == 3:
        data = np.reshape(data, (-1,)+data.shape)

    for img in data:
        # accept images of different sizes, standardize them
        if force_format is not None:
            # if grayscale to rgb needs special manipulation:
            if np.shape(img)[-1] == 1 and np.shape(force_format)[-1] == 3:
                img = color.gray2rgb(np.reshape(img, img.shape[0:-1]), alpha=False)
            img = transform.resize(img, force_format, mode='constant')
        # NOTE: output of resize is always in a [0, 1] float range

        # if in need of different format, apply a transformation
        if affine_transform is not None:
            img = img * affine_transform[0] + affine_transform[1]

        out.append(img)

    return np.array(out)
