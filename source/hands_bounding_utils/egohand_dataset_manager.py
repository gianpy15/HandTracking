from tqdm import tqdm
from skimage import io as sio
from scipy.misc import imresize
import source.hands_bounding_utils.utils as u
import os
from data_manager.path_manager import resources_path
import random
import math
import numpy as np
from scipy import io as scio
import hands_regularizer.regularizer as reg



# ######### SQUARES ############


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


def __get_path(baseim, baseannots, images, annots, number):
    dir = number // 100
    imagenum = number - dir * 100
    folderimages = os.path.join(baseim, images[dir])
    folderannots = os.path.join(baseannots, annots[dir])
    imageslist = os.listdir(folderimages)
    annotslist = os.listdir(folderannots)
    finalimagepath = os.path.join(folderimages, imageslist[imagenum])
    finalannotpath = os.path.join(folderannots, annotslist[imagenum])
    return finalimagepath, finalannotpath


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
    num = number
    if number < 0 or number > 4800:
        num = 4800
    images = []
    heatmaps = []
    for i in range(num):
        ipath, apath = __get_path(imagesfolderpath, annotationsfolderpath, images_paths, annots_paths, i)
        im = np.array(u.read_image(ipath))
        im = __add_padding(im, width_shrink_rate - im.shape[1] % width_shrink_rate,
                           height_shrink_rate - im.shape[0] % height_shrink_rate)
        images.append(np.array(im))
        heat = u.get_heatmap_from_mat(im, apath,
                                      height_shrink_rate, width_shrink_rate, overlapping_penalty, egohands=True)
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
    images = []
    heatmaps = []
    i = 0
    while i <= number:
        rand = math.floor(random.uniform(0, 4799))
        ipath, apath = __get_path(imagesfolderpath, annotationsfolderpath, images_paths, annots_paths, rand)
        im = np.array(u.read_image(ipath))
        im = __add_padding(im, width_shrink_rate - im.shape[1] % width_shrink_rate,
                           height_shrink_rate - im.shape[0] % height_shrink_rate)
        images.append(np.array(im))
        heat = u.get_heatmap_from_mat(im, apath,
                                      height_shrink_rate, width_shrink_rate, overlapping_penalty, egohands=True)
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


def default_train_annotations_path():
    return resources_path(os.path.join("hands_bounding_dataset", "egohands_squares", "annotations"))


# ############### POLYGONS ######################à


def create_dataset(videos_list=None, savepath=None, resize_rate=1.0, heigth_shrink_rate=10, width_shrink_rate=10,
                   im_reg=reg.Regularizer(), heat_reg=reg.Regularizer()):
    """reads the videos specified as parameter and for each frame produces and saves a .mat file containing
    the frame, the corresponding heatmap indicating the position of the hand(s)
    THE PROCESS:
        - image is resized
        - heatmap is produced with dimensions resized w.r.t. the resized image
        - regularizers are applied (BE CAREFUL WITH REGULARIZERS, YOU MAY ALTER DIMENSION RATIOS BETWEEN IMAGES AND
            HEATMAPS)
    :param width_shrink_rate: shrink rate of heatmaps width wrt the resized image
    :param heigth_shrink_rate: shrink rate of heatmaps height wrt the resized image
    :param resize_rate: resize rate of the images (1 (default) is the original image)
    :param savepath: path of the folder where the produces .mat files will be saved. If left to the default value None,
    the /resources/hands_bounding_dataset/egohands_transformed folder will be used
    :param videos_list: list of videos you need the .mat files of. If left to the default value None, all videos will
    be exploited
    :param im_reg: object used to regularize the images
    :param heat_reg: object used to regularize the heatmaps"""
    if savepath is None:
        basedir = resources_path(os.path.join("hands_bounding_dataset", "egohands_tranformed"))
    else:
        basedir = savepath
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    framesdir = resources_path(os.path.join("hands_bounding_dataset", "egohands"))
    if videos_list is None:
        vids = os.listdir(framesdir)
        vids = [x for x in vids if os.path.isdir(os.path.join(framesdir, x))]
    else:
        vids = videos_list
    for vid in tqdm(vids):
        frames, labels = __load_egohand_video(os.path.join(framesdir, vid))
        fr_num = len(frames)
        for i in tqdm(range(0, fr_num)):
            try:
                fr_to_save = {}
                frame = frames[i]
                frame = imresize(frame, resize_rate)
                label = labels[i]
                label *= resize_rate
                label = np.array(label, dtype=np.int32)
                frame = __add_padding(frame, frame.shape[1] - (frame.shape[1]//width_shrink_rate)*width_shrink_rate,
                                      frame.shape[0] - (frame.shape[0] // heigth_shrink_rate) * heigth_shrink_rate)
                heat = __create_ego_heatmap(frame, label, heigth_shrink_rate, width_shrink_rate)
                frame = im_reg.apply(frame)
                heat = heat_reg.apply(heat)
                fr_to_save['frame'] = frame
                fr_to_save['heatmap'] = __heatmap_to_uint8(heat)
                path = os.path.join(basedir, vid + "_" + str(i))
                scio.savemat(path, fr_to_save)
            except ValueError as e:
                print(vid + str(i) + " => " + e)


def __create_ego_heatmap(frame, label, heigth_shrink_rate, width_shrink_rate):
    pass


def __load_egohand_video(dir):
    images = os.listdir(dir)
    images.remove('polygons.mat')
    labels = __load_egohand_mat(os.path.join(dir, 'polygons.mat'))
    frames = []
    n = len(images)
    for i in range(n):
        frames.append(sio.imread(os.path.join(dir, images[i])))
    return np.array(frames), labels


def __load_egohand_mat(filepath):
    mat_cont = scio.loadmat(filepath)
    labels = []
    n = len(mat_cont['polygons']['myleft'][0])
    for i in range(n):
        single_lab = []
        single_lab.append(mat_cont['polygons']['myleft'][0][i])
        single_lab.append(mat_cont['polygons']['myright'][0][i])
        single_lab.append(mat_cont['polygons']['yourleft'][0][i])
        single_lab.append(mat_cont['polygons']['yourright'][0][i])
        labels.append(single_lab)
    return labels


def __heatmap_to_uint8(heat):
    heat = heat * 255
    heat.astype(np.uint8, copy=False)
    return heat


def __heatmap_uint8_to_float32(heat):
    heat.astype(np.float32, copy=False)
    heat = heat / 255
    return heat


if __name__ == '__main__':
    __load_egohand_mat(resources_path(os.path.join("hands_bounding_dataset", "egohands", "CARDS_COURTYARD_B_T"
                                                      , "polygons.mat")))