# This file encloses functions to read/write data to/from files of different formats.

import scipy.io
import os
import numpy as np
from skimage import transform
from skimage import color
from matplotlib import image
import re


def matrix_loader(filename, field_name='X_red'):
    """
    This method loads a matrix from a matlab .mat file
    :param filename: the path to the file
    :param field_name: is the regex matching the fields on the mat file
    :return: an array of matrices corresponding to matching fields in the .mat file
    """
    matrix = scipy.io.loadmat(filename)
    keys = [k for k in matrix.keys() if re.compile(field_name).match(k)]
    ret = []
    for k in keys:
        ret.extend(matrix[k])
    return np.array(ret)


def save_mat(filename, **variables):
    """
    Save a data dictionary to a new .mat file
    :param filename: the path to the new .mat to be created
    :param variables: a dictionary name: value to be saved into the .mat
    """
    scipy.io.savemat(filename, variables)


def save_image_from_matrix(matrix, path):
    """
    This method saves an image from a 3D RGB matrix
    :param matrix: the matrix representing image pixels
    :param path: path for the png image
    """
    image.imsave(path, matrix)


def load_from_png(path):
    """
    This method loads an image from a 3D RGB matrix
    :param path: path for the png image
    :return: the image as a matrix of floating point values
    """
    matrix = image.imread(path)
    return 1.0 * matrix


def load(path, field_name=None, force_format=None, affine_transform=None, alpha=False):
    """
    Load images of all supported formats (currently .mat, .jpg, .png) from all paths
    specified, and return them as a unique batch of normalized images.
    :param path: a list of paths with arbitrary nesting
    :param field_name: in case .mat files are found, this field name regex is used to access all of them
    :param force_format: forces all images to have a defined shape and size, regardless of their original shape.
                        use: specify a list of dimensions [height, width, channels], or None to preserve original format
                        note: if not None output images are expressed in [0.0, 1.0]
                                floating point range, original otherwise
    :param affine_transform: apply on each output image:
                img = img * affine_transform[0] + affine_transform[1]

    :param alpha: decide whether to keep alpha channel (bool)
    :return: the batch of loaded images
    """
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
            newformat = [force_format[idx] if force_format[idx] is not None else img.shape[idx]
                         for idx in range(len(force_format))]
            img = transform.resize(img, newformat, mode='constant')
            if np.max(img) > 1.0:
                img /= 255.0

        # if in need of different format, apply a transformation
        if affine_transform is not None:
            img = img * affine_transform[0] + affine_transform[1]

        out.append(img)

    return np.array(out)
