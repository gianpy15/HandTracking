import random
import os
import hands_bounding_utils.utils as u
import numpy as np
import source.hand_data_management.video_loader as vl
import hands_regularizer.regularizer as reg
import tqdm
import scipy.io as scio
from neural_network.keras.utils.naming import *
import pandas as pd

RIGHT = 1
LEFT = 0
csv_path = resources_path(os.path.join("csv", "left_right.csv"))

def load_labelled_videos(vname, getdepth=False, fillgaps=False, gapflags=False, verbosity=0):
    """given a video name, returns some information on its frames and their labels, basing on the parameters
    :param verbosity: set to True to see some prints
    :param gapflags: STILL NOT IMPLEMENTED
    :param fillgaps: is this is true, interpolated frames will be returned as well
    :param getdepth: if this is true, the method will return depths and labels, if this is false, the method
    will return frames and labels
    :param vname: name of the video. Note that the video must be present in the framedata folder, under resources
    """
    frames, labels = vl.load_labeled_video(vname, getdepth, fillgaps, gapflags)
    frames = np.array(frames)
    labels = np.array(labels)
    if verbosity == 1:
        print("FRAMES SHAPE: ", frames.shape)
        print("LABELS SHAPE: ", labels.shape)
    return frames, labels


def create_dataset(videos_list=None, savepath=None, im_regularizer=reg.Regularizer(),
                   fillgaps=False, enlarge=0.5, data_augment=False):
    """reads the videos specified as parameter and for each frame produces and saves a .mat file containing
    the frame, the corresponding heatmap indicating the position of the hand and the modified depth.
    :param fillgaps: set to True to also get interpolated frames
    :param im_regularizer: object used to regularize the cuts
    :param savepath: path of the folder where the produces .mat files will be saved. If left to the default value None,
    the /resources/hands_bounding_dataset/hands_rgbd_transformed folder will be used
    :param videos_list: list of videos you need the .mat files of. If left to the default value None, all videos will
    be exploited
    :param enlarge: crops enlarge factor"""
    if savepath is None:
        basedir = resources_path("left_right_classification_dataset")
    else:
        basedir = savepath
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    framesdir = resources_path("framedata")
    if videos_list is None:
        vids = os.listdir(framesdir)
        vids = [x for x in vids if os.path.isdir(os.path.join(framesdir, x))]
    else:
        vids = videos_list
    for vid in tqdm.tqdm(vids):
        frames, labels = load_labelled_videos(vid, fillgaps=fillgaps)
        fr_num = frames.shape[0]
        result = get_right_left(vid)
        for i in tqdm.tqdm(range(0, fr_num)):
            if labels[i] is not None:
                try:
                    frame = frames[i]
                    label = labels[i][:, 0:2]
                    label *= [frame.shape[1], frame.shape[0]]
                    label = np.array(label, dtype=np.int32).tolist()
                    label = [[p[1], p[0]] for p in label]
                    coords = __get_coord_from_labels(label)
                    cut = u.crop_from_coords(frame, coords, enlarge)
                    cut = im_regularizer.apply(cut)
                    path = os.path.join(basedir, vid + "_" + str(i))
                    __persist_frame(path, cut, result)
                    if data_augment:
                        path = os.path.join(basedir, vid + "_t_" + str(i))
                        transp = cut.squeeze().transpose()
                        transp = transp.reshape([transp.shape[0], transp.shape[1], 1])
                        __persist_frame(path, transp, 1 - result)
                except ValueError as e:
                    print("Error " + str(e) + " on vid " + vid + str(i))


def read_dataset(path=None, verbosity=0, leave_out=None):
    """reads the .mat files present at the specified path. Note that those .mat files MUST be created using
    the create_dataset method
    :param verbosity: setting this parameter to True will make the method print the number of .mat files read
    every time it reads one
    :param path: path where the .mat files will be looked for. If left to its default value of None, the default path
    /resources/hands_bounding_dataset/hands_rgbd_transformed folder will be used
    :param leave_out: list of videos whose elements will be put in the test set. Note that is this parameter is not
    provided, only 2 arrays will be returned (frames, labels). If this is provided, 4 arrays are returned
    (frames, labels, test_frames, test_labels)
    """
    if path is None:
        basedir = resources_path("left_right_classification_dataset")
    else:
        basedir = path
    samples = os.listdir(basedir)
    i = 0
    tot = len(samples)
    frames = []
    label = []
    t_frames = []
    t_label = []
    for name in samples:
        if verbosity == 1:
            print("Reading image: ", i, " of ", tot)
            i += 1
        realpath = os.path.join(basedir, name)
        readframes, readrl = __read_frame(realpath)
        if leave_out is None or not __matches(name, leave_out):
            frames.append(readframes)
            label.append(readrl)
        else:
            t_frames.append(readframes)
            t_label.append(readrl)
    if leave_out is None:
        return frames, label
    return frames, label, t_frames, t_label


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
        basedir = resources_path("left_right_classification_dataset")
    else:
        basedir = path
    samples = os.listdir(basedir)
    if leave_out is not None:
        samples = [s for s in samples if not __matches(s, leave_out)]
    tot = len(samples)
    if number > tot:
        raise ValueError("number must be smaller than the number of samples")
    frames = []
    labels = []
    for i in range(number):
        if verbosity == 1:
            print("Reading image: ", i, " of ", tot)
            i += 1
        which = int(np.math.floor(random.uniform(0, tot - 0.01)))
        realpath = os.path.join(basedir, samples[which])
        samples.pop(which)
        tot -= 1
        readcuts, readlabels = __read_frame(realpath)
        frames.append(readcuts)
        labels.append(readlabels)
    return frames, labels


def shuffle_cut_label(ri, di):
    n = np.shape(ri)[0]
    pos = list(range(0, n))
    n -= 0.01
    r1 = []
    d1 = []
    while n > 0:
        rand = int(np.math.floor(random.uniform(0, n)))
        r1.append(ri[pos[rand]])
        d1.append(di[pos[rand]])
        pos.pop(rand)
        n -= 1
    return np.array(r1), np.array(d1)


def __read_frame(path):
    matcontent = scio.loadmat(path)
    return matcontent['cut'], matcontent['rl'][0][0]


def get_right_left(vid):
    csv_data = pd.read_csv(csv_path)
    num = len(csv_data.index)
    for i in range(num):
        row = csv_data.iloc[i]
        if row['vidname'] == vid:
            return row['lab']
    return RIGHT



def __matches(s, leave_out):
    for stri in leave_out:
        if s.startswith(stri + "_"):
            return True
    return False


def __get_coord_from_labels(lista):
    list_x = np.array([p[0] for p in lista])
    list_y = np.array([p[1] for p in lista])
    min_x = np.min(list_x)
    max_x = np.max(list_x)
    min_y = np.min(list_y)
    max_y = np.max(list_y)
    return [[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]


def __persist_frame(path, cut, rl):
    fr_to_save = {'cut': cut,
                  'rl': rl}
    scio.savemat(path, fr_to_save)


def count_ones_zeros(y_train, y_test):
    right = 0
    left = 0
    for i in range(len(y_train)):
        if y_train[i] == 1:
            right += 1
        else:
            left += 1
    print("TRAIN R: ", right)
    print("TRAIN L: ", left)
    right = 0
    left = 0
    for i in range(len(y_test)):
        if y_test[i] == 1:
            right += 1
        else:
            left += 1
    print("TEST R: ", right)
    print("TEST L: ", left)


if __name__ == '__main__':
    im_r = reg.Regularizer()
    im_r.fixresize(200, 200)
    im_r.rgb2gray()
    #create_dataset(im_regularizer=im_r)
    c, b = read_dataset_random()
    print(b[0])
    u.showimage(c[0].squeeze())

