import random
import data.datasets.crop.utils as u
import numpy as np
import data.datasets.framedata_management.video_loader as vl
import data.regularization.regularizer as reg
import tqdm
import scipy.io as scio
from data.naming import *
from library.geometry.left_right_detection import palmback as pb
from library.geometry.formatting import hand_format
import pandas as pd
from library.geometry.formatting import *
from library.geometry.left_right_detection.palmback import leftright_to_palmback

RIGHT = 1
LEFT = 0
csv_path = resources_path("csv", "left_right.csv")

PALM = 1.0
BACK = -1.0


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
                   fillgaps=False, enlarge=0.5):
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
        basedir = resources_path("palm_back_classification_dataset")
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
        lr = get_right_left(vid)
        if lr == LEFT:
            lr = -1
        for i in tqdm.tqdm(range(0, fr_num)):
            if labels[i] is not None:
                try:
                    frame = frames[i]
                    label = labels[i][:, 0:2]

                    # conf is a real in [-1.0, 1.0] such that -1.0 is full back, +1.0 is full palm
                    # middle values express partial confidence, but all info is in one single value
                    conf = pb.leftright_to_palmback(hand=hand_format(label),
                                                    side=pb.RIGHT if lr == RIGHT else pb.LEFT)
                    # if you want the crisp result, there it is:
                    result = PALM if conf >= 0 else BACK
                    # and the confidence on that result is in [0..1]:
                    conf = abs(conf)

                    label *= [frame.shape[1], frame.shape[0]]
                    label = np.array(label, dtype=np.int32).tolist()
                    label = [[p[1], p[0]] for p in label]
                    coords = __get_coord_from_labels(label)
                    cut = u.crop_from_coords(frame, coords, enlarge)
                    cut = im_regularizer.apply(cut)
                    path = os.path.join(basedir, vid + "_" + str(i))
                    __persist_frame(path, cut, result, conf)
                except ValueError as e:
                    print("Error " + str(e) + " on vid " + vid + str(i))


def get_palm_back(label, lr):
    label = hand_format(label)
    res = leftright_to_palmback(label, lr)
    result = 1
    if res < 0:
        result = 0
    conf = abs(res)
    return result, conf



def read_dataset(path=None, verbosity=0, leave_out=None):
    """reads the .mat files present at the specified path. Note that those .mat files MUST be created using
    the create_dataset method
    :param verbosity: setting this parameter to True will make the method print the number of .mat files read
    every time it reads one
    :param path: path where the .mat files will be looked for. If left to its default value of None, the default path
    /resources/hands_bounding_dataset/hands_rgbd_transformed folder will be used
    :param leave_out: list of videos whose elements will be put in the test set. Note that is this parameter is not
    provided, only 3 arrays will be returned (frames, labels, conf). If this is provided, 6 arrays are returned
    (frames, labels, conf, test_frames, test_labels, test_conf)
    """
    if path is None:
        basedir = resources_path("palm_back_classification_dataset")
    else:
        basedir = path
    samples = os.listdir(basedir)
    i = 0
    tot = len(samples)
    frames = []
    label = []
    conf = []
    t_frames = []
    t_label = []
    t_conf = []
    for name in samples:
        if verbosity == 1:
            print("Reading image: ", i, " of ", tot)
            i += 1
        realpath = os.path.join(basedir, name)
        readframes, readrl, readconf = __read_frame(realpath)
        if leave_out is None or not __matches(name, leave_out):
            frames.append(readframes)
            label.append(readrl)
            conf.append(readconf)
        else:
            t_frames.append(readframes)
            t_label.append(readrl)
            t_conf.append(readconf)
    if leave_out is None:
        return frames, label, conf
    return frames, label, conf, t_frames, t_label, t_conf


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
        basedir = resources_path("palm_back_classification_dataset")
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
    conf = []
    for i in range(number):
        if verbosity == 1:
            print("Reading image: ", i, " of ", tot)
            i += 1
        which = int(np.math.floor(random.uniform(0, tot - 0.01)))
        realpath = os.path.join(basedir, samples[which])
        samples.pop(which)
        tot -= 1
        readcuts, readlabels, rconf = __read_frame(realpath)
        frames.append(readcuts)
        labels.append(readlabels)
        conf.append(rconf)
    return frames, labels, conf


def shuffle_cut_label_conf(ri, di, ci):
    n = np.shape(ri)[0]
    pos = list(range(0, n))
    n -= 0.01
    r1 = []
    d1 = []
    c1 = []
    while n > 0:
        rand = int(np.math.floor(random.uniform(0, n)))
        r1.append(ri[pos[rand]])
        d1.append(di[pos[rand]])
        c1.append(ci[pos[rand]])
        pos.pop(rand)
        n -= 1
    return np.array(r1), np.array(d1), np.array(c1)


def __read_frame(path):
    matcontent = scio.loadmat(path)
    return matcontent['cut'], matcontent['pb'][0][0], matcontent['conf'][0][0]


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


def __persist_frame(path, cut, pb, conf):
    fr_to_save = {'cut': cut,
                  'pb': pb,
                  'conf': conf}
    scio.savemat(path, fr_to_save)


def attach_out_conf(y, c):
    n = len(y)
    ris = []
    for i in range(n):
        ris.append([y[i], c[i]])
    return np.array(ris)


if __name__ == '__main__':
    im_r = reg.Regularizer()
    im_r.fixresize(200, 200)
    im_r.rgb2gray()
    #create_dataset(im_regularizer=im_r)
    c, r, co = read_dataset_random()
    print(r[0], co[0])
    u.showimage(c[0].squeeze())

