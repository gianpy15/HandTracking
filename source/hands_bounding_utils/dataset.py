import source.hands_bounding_utils.utils as u
import os
from data_manager import path_manager
import random
import math
import numpy as np
pm = path_manager.PathManager()


def __list_objects_in_path(path):
    """lists all the objects at the given path"""
    return os.listdir(path)


def __clean_list(lista):
    """removes '__init__.py' from the given list and returns the clean list"""
    lista.remove('__init__.py')
    return lista


def __add_padding(image, right_pad, bottom_pad):
    image = np.hstack((image, np.zeros([image.shape[0], right_pad, 3], dtype=np.uint8)))
    image = np.vstack((image, np.zeros([bottom_pad, image.shape[1], 3], dtype=np.uint8)))
    return image


def get_samples_from_dataset_in_order_from_beginning(imagesfolderpath, annotationsfolderpath, number,
                                                     height_shrink_rate=10, width_shrink_rate=10,
                                                     overlapping_penalty=0.9):
    """gets a number of samples from the dataset that will be used to train the network that locates hands from images,
    as first step of the computation of the 3D model of the hands
    :type imagesfolderpath: the path of THE FOLDER that contains the images.
    :type annotationsfolderpath: the path of the FOLDER that contains annotations.
    :type number: the number of images to get. If this is <=0 or >=Total_number_of_images, all images will be
    returned.
    :type height_shrink_rate: the factor that represents how much the height of heatmaps will be shrunk w.r.t.
    the original image
    :type width_shrink_rate: the factor that represents how much the width of heatmaps will be shrunk w.r.t.
    the original image
    :type overlapping_penalty: how much overlapping of hands is penalized in the heatmaps
    Note that for this function to work properly, annotations and images must be in DIFFERENT folders and must have
    THE SAME NAME, except for the extension"""
    images_paths = __clean_list(__list_objects_in_path(imagesfolderpath))
    annots_paths = __clean_list(__list_objects_in_path(annotationsfolderpath))
    num = len(images_paths)
    if 0 < number < num:
        num = number
    images = []
    heatmaps = []
    for i in range(num):
        im = np.array(u.read_image(os.path.join(imagesfolderpath, images_paths[i])))
        im = __add_padding(im, width_shrink_rate - im.shape[1] % width_shrink_rate,
                           height_shrink_rate - im.shape[0] % height_shrink_rate)
        images.append(np.array(im))
        heat = u.get_heatmap_from_mat(im, os.path.join(annotationsfolderpath, annots_paths[i]),
                                      height_shrink_rate, width_shrink_rate, overlapping_penalty)
        heat = u.heatmap_to_3d(heat)
        heatmaps.append(np.array(heat))
    return images, heatmaps


def get_random_samples_from_dataset(imagesfolderpath, annotationsfolderpath, number,
                                                     height_shrink_rate=10, width_shrink_rate=10,
                                                     overlapping_penalty=0.9):
    """gets a number of random samples from the dataset that will be used to train the network that locates hands from
    images, as first step of the computation of the 3D model of the hands
    :type imagesfolderpath: the path of THE FOLDER that contains the images.
    :type annotationsfolderpath: the path of the FOLDER that contains annotations.
    :type number: the number of images to get. If this is <=0 or >=Total_number_of_images, all images will be
    returned.
    :type height_shrink_rate: the factor that represents how much the height of heatmaps will be shrunk w.r.t.
    the original image
    :type width_shrink_rate: the factor that represents how much the width of heatmaps will be shrunk w.r.t.
    the original image
    :type overlapping_penalty: how much overlapping of hands is penalized in the heatmaps
    Note that for this function to work properly, annotations and images must be in DIFFERENT folders and must have
    THE SAME NAME, except for the extension"""
    images_paths = __clean_list(__list_objects_in_path(imagesfolderpath))
    annots_paths = __clean_list(__list_objects_in_path(annotationsfolderpath))
    num = len(images_paths)
    images = []
    heatmaps = []
    i = 0
    while i <= number:
        rand = math.floor(random.uniform(0, num))
        im = np.array(u.read_image(os.path.join(imagesfolderpath, images_paths[rand])))
        im = __add_padding(im, width_shrink_rate - im.shape[1] % width_shrink_rate,
                           height_shrink_rate - im.shape[0] % height_shrink_rate)
        images.append(np.array(im))
        heat = u.get_heatmap_from_mat(im, os.path.join(annotationsfolderpath, annots_paths[rand]),
                                      height_shrink_rate, width_shrink_rate, overlapping_penalty)
        heat = u.heatmap_to_3d(heat)
        heatmaps.append(np.array(heat))
        i += 1
    return images, heatmaps


def get_ordered_batch(images, heatmaps, batch_size, batch_number):
    """from the given sets "images" and "heatmaps", returns a batch of size batch_size.
    The images are picked in order, starting from batch_size*batch_number. It will return less elements if
    the end of the list is reached before."""
    real_size = batch_size
    if batch_size*(batch_number+1) > len(images):
        real_size = batch_size*(batch_number+1) - len(images) + 1
    start = batch_size*batch_number
    end = start + real_size
    return images[start:end], heatmaps[start:end]


def get_random_batch(images, heatmaps, batch_size):
    """returns a random batch of size batch_size. The elements are taken from the given sets images and heatmaps."""
    size = 0
    tot = len(images)
    ims = []
    heats = []
    while size < batch_size:
        rand = math.floor(random.uniform(0, tot))
        ims.append(images[rand])
        heats.append(heatmaps[rand])
        size += 1
    return ims, heats


if __name__ == '__main__':
    im_f = pm.resources_path("hands_bounding_dataset/train/images")
    an_f = pm.resources_path("hands_bounding_dataset/train/annotations")
    images1, heatmaps1 = get_samples_from_dataset_in_order_from_beginning(im_f, an_f, 100)
    # images1, heatmaps1 = get_ordered_batch(images1, heatmaps1, 1, 1)
    images1, heatmaps1 = get_random_batch(images1, heatmaps1, 2)
    u.showimages(images1)
    for heat1 in heatmaps1:
        u.showimage(u.heatmap_to_rgb(heat1))
