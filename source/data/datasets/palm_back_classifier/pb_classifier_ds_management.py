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
from library.utils.deprecation import deprecated_fun
from data.datasets.reading.dataset_manager import DatasetManager

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
                    if result < 0:
                        result = 0
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


def create_dataset_w_heatmaps(videos_list=None, savepath=None, im_regularizer=reg.Regularizer(), h_r=reg.Regularizer(),
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
        basedir = resources_path("palm_back_classification_dataset_h")
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
                    if result < 0:
                        result = 0
                    # and the confidence on that result is in [0..1]:
                    conf = abs(conf)
                    heat = np.zeros([frame.shape[0], frame.shape[1]])
                    label *= [frame.shape[1], frame.shape[0]]
                    label = np.array(label, dtype=np.int32).tolist()
                    label = [[p[1], p[0]] for p in label]
                    coords = __get_coord_from_labels(label)
                    heat = __shade_heatmap(heat, coords, label)
                    heat = u.crop_from_coords(heat, coords, enlarge)
                    heat = h_r.apply(heat)
                    heat.reshape([heat.shape[0], heat.shape[1], 1])
                    cut = u.crop_from_coords(frame, coords, enlarge)
                    cut = im_regularizer.apply(cut)
                    path = os.path.join(basedir, vid + "_" + str(i))
                    __persist_frame_h(path, cut, result, conf, heat)
                except ValueError as e:
                    print("Error " + str(e) + " on vid " + vid + str(i))


def __shade_heatmap(heat, square_coords, joint_coords):
    h = [s[0] for s in square_coords]
    w = [w[1] for w in square_coords]
    min_h = np.min(h)
    max_h = np.max(h)
    min_w = np.min(w)
    max_w = np.max(w)
    supp_heat = np.zeros([max_h + 1 - min_h, max_w + 1 - min_w])
    for i in range(min_h, max_h + 1):
        for j in range(min_w, max_w + 1):
            if __in_triangle([i, j], joint_coords[0], joint_coords[5], joint_coords[17]) or \
                    __in_triangle([i, j], joint_coords[1], joint_coords[5], joint_coords[17]) or \
                    __in_triangle([i, j], joint_coords[0], joint_coords[2], joint_coords[17]):
                supp_heat[i - min_h][j - min_w] = 0
            else:
                distances = [__dist([i, j], jc) for jc in joint_coords]
                distances2 = __dists_p_seg([i, j], joint_coords)
                supp_heat[i-min_h][j-min_w] = np.min(distances + distances2)
    mean = np.mean(supp_heat)
    std = np.std(supp_heat)
    heat[min_h:max_h + 1, min_w: max_w + 1] = (supp_heat - mean) / std
    heat[min_h:max_h + 1, min_w: max_w + 1] -= np.min(heat[min_h:max_h + 1, min_w: max_w + 1])
    min_gauss = __percentile(heat[min_h:max_h + 1, min_w: max_w + 1], - 0.1 + __sigmoid(mean/5))
    for i in range(min_h, max_h + 1):
        for j in range(min_w, max_w + 1):
            if heat[i][j] > min_gauss:
                heat[i][j] = 0.
            elif min_gauss != 0:
                heat[i][j] = np.math.sqrt(1 - (heat[i][j] * heat[i][j])/(min_gauss * min_gauss))
            else:
                heat[i][j] = 1.

    return heat


def __sigmoid(x):
    return 1 / (1 + np.math.exp(-x))


def __in_triangle(p, v0, v1, v2):
    v1 = [v1[0] - v0[0], v1[1] - v0[1]]
    v2 = [v2[0] - v0[0], v2[1] - v0[1]]
    vv2 = [p, v2]
    vv1 = [p, v1]
    v0v2 = [v0, v2]
    v1v2 = [v1, v2]
    v0v1 = [v0, v1]
    try:
        a = (__det22(vv2) - __det22(v0v2)) / __det22(v1v2)
        b = - (__det22(vv1) - __det22(v0v1)) / __det22(v1v2)
        return a > 0 and b > 0 and a + b <= 1
    except ZeroDivisionError:
        return False


def __det22(mat):
    return mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1]


def __percentile(mat, perc):
    array = mat.flatten()
    array = list(array)
    array = sorted(array)
    return array[int(len(array)*perc)]


def __dists_p_seg(p, joints):
    ris = []
    num = len(joints)
    for i in range(num - 1):
        if i % 4 != 0 or i == 0:
            ris.append(__dist_p_seg(p, joints[i], joints[i+1]))
    ris.append(__dist_p_seg(p, joints[0], joints[5]))
    ris.append(__dist_p_seg(p, joints[0], joints[9]))
    ris.append(__dist_p_seg(p, joints[0], joints[13]))
    ris.append(__dist_p_seg(p, joints[0], joints[17]))
    ris.append(__dist_p_seg(p, joints[1], joints[5]))
    ris.append(__dist_p_seg(p, joints[1], joints[9]))
    ris.append(__dist_p_seg(p, joints[1], joints[13]))
    ris.append(__dist_p_seg(p, joints[1], joints[17]))
    ris.append(__dist_p_seg(p, joints[1], joints[5]))
    ris.append(__dist_p_seg(p, joints[5], joints[9]))
    ris.append(__dist_p_seg(p, joints[9], joints[13]))
    ris.append(__dist_p_seg(p, joints[13], joints[17]))
    return ris


def __dist_p_seg(p, e1, e2):
    if e2[0] - e1[0] != 0 and e2[1] - e1[1] != 0:
        ang_coeff = (e2[1] - e1[1]) / (e2[0] - e1[0])
        intercept = e2[1] - ang_coeff * e2[0]
        ang_coeff1 = - 1 / ang_coeff
        ang_coeff2 = - 1 / ang_coeff
        p1 = e1
        p2 = e1
        intercept1 = e2[1] - ang_coeff1 * e2[0]
        intercept2 = e1[1] - ang_coeff2 * e1[0]
        if intercept1 > intercept2:
            p1 = e2
            p2 = e1
            app = intercept1
            intercept1 = intercept2
            intercept2 = app
        if p[1] > ang_coeff1 * p[0] + intercept1 and p[1] < ang_coeff2 * p[0] + intercept2:
            return __dist_p_rect(p, ang_coeff, intercept)
        if p[1] <= ang_coeff1 * p[0] + intercept1:
            return __dist(p, p1)
        return __dist(p, p2)
    if e2[0] - e1[0] == 0:
        p1 = e1
        p2 = e2
        if e1[1] > e2[1]:
            p2 = e1
            p1 = e2
        if p1[1] <= p[1] <= p2[1]:
            return abs(p[0] - p2[0])
        if p[1] < p1[1]:
            return __dist(p, p1)
        return __dist(p, p2)
    p1 = e1
    p2 = e2
    if e1[0] > e2[0]:
        p2 = e1
        p1 = e2
    if p1[0] <= p[0] <= p2[0]:
        return abs(p[1] - p2[1])
    if p[0] < p1[0]:
        return __dist(p, p1)
    return __dist(p, p2)


def __dist_p_rect(p, m, q):
    return abs(p[1] - m*p[0] - q) / np.math.sqrt(1 + m * m)


def __extend_joints(joints):
    num = len(joints)
    joints = np.array(joints)
    ris = []
    for i in range(num -1):
        ris.append(joints[i])
        if i % 4 != 0 or i == 0:
            ris.append((joints[i] + joints[i+1]) // 2)
        ris.append((joints[0] + joints[5]) // 2)
        ris.append((joints[0] + joints[9]) // 2)
        ris.append((joints[0] + joints[13]) // 2)
        ris.append((joints[0] + joints[17]) // 2)
    return ris


def __dist(p1, p2):
    return np.math.sqrt(np.math.pow(p1[0]-p2[0], 2) + np.math.pow(p1[1]-p2[1], 2))


def get_palm_back(label, lr):
    label = hand_format(label)
    res = leftright_to_palmback(label, lr)
    result = 1
    if res < 0:
        result = 0
    conf = abs(res)
    return result, conf

@deprecated_fun(alternative=DatasetManager)
def read_dataset(path=None, verbosity=0, leave_out=None, minconf=0.0):
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
        if readconf >= minconf:
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


@deprecated_fun(alternative=DatasetManager)
def read_dataset_h(path=None, verbosity=0, videos=None, leave_out=None, minconf=0.0):
    """reads the .mat files present at the specified path. Note that those .mat files MUST be created using
    the create_dataset method
    :param verbosity: setting this parameter to True will make the method print the number of .mat files read
    every time it reads one
    :param path: path where the .mat files will be looked for. If left to its default value of None, the default path
    /resources/hands_bounding_dataset/hands_rgbd_transformed folder will be used
    :param leave_out: list of videos whose elements will be put in the test set. Note that is this parameter is not
    provided, only 4 arrays will be returned (frames, labels, conf, h). If this is provided, 8 arrays are returned
    (frames, labels, conf, h, test_frames, test_labels, test_confh t_h)
    """
    if path is None:
        basedir = resources_path("palm_back_classification_dataset_h")
    else:
        basedir = path
    samples = os.listdir(basedir)
    if videos is not None:
        samples = [s for s in samples if __matches(s, videos)]
    i = 0
    tot = len(samples)
    frames = []
    label = []
    conf = []
    heat = []
    t_frames = []
    t_label = []
    t_conf = []
    t_heat = []
    for name in samples:
        if verbosity == 1:
            print("Reading image: ", i, " of ", tot)
            i += 1
        realpath = os.path.join(basedir, name)
        readframes, readrl, readconf, rh = __read_frame_h(realpath)
        if readconf >= minconf:
            if leave_out is None or not __matches(name, leave_out):
                frames.append(readframes)
                label.append(readrl)
                conf.append(readconf)
                heat.append(rh)
            else:
                t_frames.append(readframes)
                t_label.append(readrl)
                t_conf.append(readconf)
                t_heat.append(rh)
    if leave_out is None:
        return frames, label, conf, heat
    return frames, label, conf, heat, t_frames, t_label, t_conf, t_heat


@deprecated_fun(alternative=DatasetManager)
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


def shuffle_cut_label_conf_h(ri, di, ci, hi):
    n = np.shape(ri)[0]
    pos = list(range(0, n))
    n -= 0.01
    r1 = []
    d1 = []
    c1 = []
    h1 = []
    while n > 0:
        rand = int(np.math.floor(random.uniform(0, n)))
        r1.append(ri[pos[rand]])
        d1.append(di[pos[rand]])
        c1.append(ci[pos[rand]])
        h1.append(hi[pos[rand]])
        pos.pop(rand)
        n -= 1
    return np.array(r1), np.array(d1), np.array(c1), np.array(h1)


def __read_frame(path):
    matcontent = scio.loadmat(path)
    return matcontent['cut'], matcontent['pb'][0][0], matcontent['conf'][0][0]


def __read_frame_h(path):
    matcontent = scio.loadmat(path)
    heat = matcontent['heatmap']
    heat = heat.reshape([heat.shape[0], heat.shape[1], 1])
    return matcontent['cut'], matcontent['pb'][0][0], matcontent['conf'][0][0], heat


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


def __persist_frame_h(path, cut, pb, conf, h):
    fr_to_save = {'cut': cut,
                  'pb': pb,
                  'conf': conf,
                  'heatmap': h}
    scio.savemat(path, fr_to_save)


def attach_out_conf(y, c):
    n = len(y)
    ris = []
    for i in range(n):
        ris.append([y[i], c[i]])
    return np.array(ris)


def count_ones_zeros(y_train, y_test):
    right = 0
    left = 0
    for i in range(len(y_train)):
        if y_train[i] == 1:
            right += 1
        else:
            left += 1
    print("TRAIN P: ", right)
    print("TRAIN B: ", left)
    class_weight={1: (left+right)/right, 0: (left+right)/left}
    right = 0
    left = 0
    for i in range(len(y_test)):
        if y_test[i] == 1:
            right += 1
        else:
            left += 1
    print("TEST P: ", right)
    print("TEST B: ", left)
    return class_weight



if __name__ == '__main__':
    im_r = reg.Regularizer()
    im_r.fixresize(200, 200)
    h_r = reg.Regularizer()
    h_r.fixresize(200, 200)
    #create_dataset_w_heatmaps(im_regularizer=im_r, h_r=h_r)
    c, r, co, h = read_dataset_h()
    print(r[1], co[1])
    u.showimage(c[1].squeeze())
    u.showimage(u.heatmap_to_rgb(h[1]))

