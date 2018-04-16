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
from geometry import polygon_utils as poly
from numba import jit
from time import time as t

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
                   im_reg=reg.Regularizer(), heat_reg=reg.Regularizer(), approximation_ratio=0.1):
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
        frames, labels = __load_egohand_video(os.path.join(framesdir, vid),
                                              one_out_of=int(min(width_shrink_rate, heigth_shrink_rate))/resize_rate)
        fr_num = len(frames)
        for i in tqdm(range(0, fr_num)):
                fr_to_save = {}
                frame = frames[i]
                frame = imresize(frame, [480, 640])
                frame = imresize(frame, resize_rate)
                frame = __add_padding(frame, frame.shape[1] - (frame.shape[1]//width_shrink_rate)*width_shrink_rate,
                                      frame.shape[0] - (frame.shape[0] // heigth_shrink_rate) * heigth_shrink_rate)
                heat = __create_ego_heatmap(frame, labels[i], heigth_shrink_rate, width_shrink_rate,
                                            resize_rate, approximation_ratio)
                frame = im_reg.apply(frame)
                heat = heat_reg.apply(heat)
                fr_to_save['frame'] = frame
                fr_to_save['heatmap'] = __heatmap_to_uint8(heat)
                path = os.path.join(basedir, vid + "_" + str(i))
                scio.savemat(path, fr_to_save)


def read_dataset(path=None, verbosity=0, leave_out=None):
    """reads the .mat files present at the specified path. Note that those .mat files MUST be created using
    the create_dataset method
    :param verbosity: setting this parameter to True will make the method print the number of .mat files read
    every time it reads one
    :param path: path where the .mat files will be looked for. If left to its default value of None, the default path
    /resources/hands_bounding_dataset/hands_rgbd_transformed folder will be used
    :param leave_out: list of videos whose elements will be put in the test set. Note that is this parameter is not
    provided, only 2 arrays will be returned (frames, heatmaps). If this is provided, 4 arrays are returned
    (frames, heatmaps, test_frames, test_heatmaps)
    """
    if path is None:
        basedir = resources_path(os.path.join("hands_bounding_dataset", "egohands_tranformed"))
    else:
        basedir = path
    samples = os.listdir(basedir)
    i = 0
    tot = len(samples)
    frames = []
    heatmaps = []
    t_frames = []
    t_heatmaps = []
    for name in samples:
        if verbosity == 1:
            print("Reading image: ", i, " of ", tot)
            i += 1
        realpath = os.path.join(basedir, name)
        readframes, readheats = __read_frame(realpath)
        if leave_out is None or not __matches(name, leave_out):
            frames.append(readframes)
            heatmaps.append(readheats)
        else:
            t_frames.append(readframes)
            t_heatmaps.append(readheats)
    if leave_out is None:
        return frames, heatmaps
    return frames, heatmaps, t_frames, t_heatmaps


def read_dataset_random(path=None, number=1, verbosity=0, leave_out=None):
    """reads "number" different random .mat files present at the specified path. Note that those .mat files MUST be created using
    the create_dataset method
    :param verbosity: setting this parameter to 1 will make the method print the number of .mat files read
    every time it reads one
    :param path: path where the .mat files will be looked for. If left to its default value of None, the default path
    /resources/hands_bounding_dataset/hands_rgbd_transformed folder will be used
    :param number: number of elements to read
    :param leave_out: list of videos from which samples will NOT be taken
    """
    if path is None:
        basedir = resources_path(os.path.join("hands_bounding_dataset", "egohands_tranformed"))
    else:
        basedir = path
    samples = os.listdir(basedir)
    if leave_out is not None:
        samples = [s for s in samples if not __matches(s, leave_out)]
    tot = len(samples)
    if number > tot:
        raise ValueError("number must be smaller than the number of samples")
    frames = []
    heatmaps = []
    for i in range(number):
        if verbosity == 1:
            print("Reading image: ", i, " of ", tot)
            i += 1
        which = int(np.math.floor(random.uniform(0, tot - 0.01)))
        realpath = os.path.join(basedir, samples[which])
        samples.pop(which)
        tot -= 1
        readcuts, readheats = __read_frame(realpath)
        frames.append(readcuts)
        heatmaps.append(readheats)
    return frames, heatmaps


def __matches(s, leave_out):
    for stri in leave_out:
        if s.startswith(stri + "_"):
            return True
    return False


def __read_frame(path):
    matcontent = scio.loadmat(path)
    return matcontent['frame'], __heatmap_uint8_to_float32(matcontent['heatmap'])


def __create_ego_heatmap(frame, label, heigth_shrink_rate, width_shrink_rate, resize_rate, approx):
    newlab = [[[p[1] * resize_rate * 480/720, p[0] * resize_rate * 640/1280] for p in l] for l in label]
    newlab = [[[p[0] // heigth_shrink_rate, p[1] // width_shrink_rate] for p in l] for l in newlab]
    newlab = [np.array(lab, dtype=np.int32) for lab in newlab]
    heat = np.zeros([int(frame.shape[0] / heigth_shrink_rate), int(frame.shape[1] / width_shrink_rate)])
    squares = __squares_from_labels(newlab)
    for i in range(len(squares)):
        square = squares[i]
        up, down, left, right = __get_bounds(square)
        h = up

        h_incr = int(1 + (down - up) * approx)
        j_incr = int(1 + (right - left) * approx)
        while up <= h <= down-1:
            j = left
            while left <= j <= right-1:
                if poly.fast_is_inside([h, j], newlab[i]):
                    end_h = h + h_incr
                    end_j = j + j_incr
                    if end_h > heat.shape[0]:
                        end_h = heat.shape[0]
                    if end_j > heat.shape[1]:
                        end_j = heat.shape[1]
                    heat[h:end_h, j:end_j] = 1
                j += j_incr
            h += h_incr
    return heat


def __get_bounds(coord):
    """given an array of 4 coordinates (x,y), simply computes and
    returns the highest and lowest vertical and horizontal points"""
    if len(coord) != 4:
        raise AttributeError("coord must be a set of 4 coordinates")
    x = [c[0] for c in coord]
    y = [c[1] for c in coord]
    up = np.min(x)
    down = np.max(x)
    left = np.min(y)
    right = np.max(y)
    return up, down, left, right


def __get_coord_from_labels(lista):
    list_x = [p[0] for p in lista]
    list_y = [p[1] for p in lista]
    min_x = np.min(list_x)
    max_x = np.max(list_x)
    min_y = np.min(list_y)
    max_y = np.max(list_y)
    return [[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]


def __squares_from_labels(label):
    squares = []
    for l in label:
        squares.append(__get_coord_from_labels(l))
    return squares


def __load_egohand_video(dir, one_out_of=10):
    images = os.listdir(dir)
    images.remove('polygons.mat')
    labels = __load_egohand_mat(os.path.join(dir, 'polygons.mat'), one_out_of)
    frames = []
    n = len(images)
    for i in range(n):
        frames.append(sio.imread(os.path.join(dir, images[i])))
    return np.array(frames), labels


def __load_egohand_mat(filepath, one_out_of=10):
    mat_cont = scio.loadmat(filepath)
    labels = []
    n = len(mat_cont['polygons']['myleft'][0])
    for i in range(n):
        single_lab = []
        single_lab.append(mat_cont['polygons']['myleft'][0][i].tolist())
        single_lab.append(mat_cont['polygons']['myright'][0][i].tolist())
        single_lab.append(mat_cont['polygons']['yourleft'][0][i].tolist())
        single_lab.append(mat_cont['polygons']['yourright'][0][i].tolist())
        for lab in single_lab:
            n = len(lab)
            j = 0
            while j < n:
                if j % one_out_of != 0:
                    lab.pop(j)
                    n -= 1
                j += 1
        labels.append(single_lab)
    for l in labels:
        try:
            while True:
                l.remove([[]])
        except ValueError:
            pass
        try:
            while True:
                l.remove([])
        except ValueError:
            pass
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
    #create_dataset(['CARDS_COURTYARD_B_T'], resize_rate=0.5, width_shrink_rate=4, heigth_shrink_rate=4)
    f, h = read_dataset_random()
    print(np.shape(f), np.shape(h))
    print(np.shape(f[0]), np.shape(h[0]))
    u.showimage(f[0])
    u.showimage(h[0])
    u.showimages(u.get_crops_from_heatmap(f[0], h[0],
                                                width_shrink_rate=4,
                                                height_shrink_rate=4,
                                                accept_crop_minimum_dimension_pixels=200))
