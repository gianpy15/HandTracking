import data.datasets.framedata_management.camera_data_conversion as cdc
import data.datasets.framedata_management.video_loader as vl
import data.datasets.framedata_management.grey_to_redblue_codec as gtrbc
import numpy as np
import tqdm
import math
import sys
from scipy import io as scio
import random
from scipy.misc import imresize
from scipy.signal import convolve
import data.datasets.crop.utils as u
from data.naming import *
from timeit import timeit as tm
import data.regularization.regularizer as reg
from library.utils.deprecation import deprecated_fun
from data.datasets.reading.dataset_manager import DatasetManager


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
    if verbosity == 1:
        print("FRAMES SHAPE: ", frames.shape)
        print("LABELS SHAPE: ", labels.shape)
    return frames, labels


def depth_resize(depth, rr):
    """used ONLY to resize the depth of frames
    :param depth: (n,m) matrix containing binary (0 or 1) values
    :type rr: resize rate
    """
    depth = depth.reshape([depth.shape[0], depth.shape[1], 1])
    depth = np.dstack((depth, depth, depth))
    return imresize(depth, rr)[:, :, 0:1]


def create_dataset(videos_list=None, savepath=crops_path(), resize_rate=1.0, heigth_shrink_rate=10, width_shrink_rate=10,
                   overlapping_penalty=0.9, fillgaps=False, toofar=1500, tooclose=500):
    """reads the videos specified as parameter and for each frame produces and saves a .mat file containing
    the frame, the corresponding heatmap indicating the position of the hand and the modified depth.
    :param tooclose: threshold value used to eliminate too close objects/values in the depth
    :param toofar: threshold value used to eliminate too far objects/values in the depth
    :param fillgaps: set to True to also get interpolated frames
    :param overlapping_penalty: penalty "inflicted" to the overlapping hands area in the images
    :param width_shrink_rate: shrink rate of heatmaps width wrt the resized image
    :param heigth_shrink_rate: shrink rate of heatmaps height wrt the resized image
    :param resize_rate: resize rate of the images (1 (default) is the original image)
    :param savepath: path of the folder where the produces .mat files will be saved. If left to the default value None,
    the /resources/hands_bounding_dataset/hands_rgbd_transformed folder will be used
    :param videos_list: list of videos you need the .mat files of. If left to the default value None, all videos will
    be exploited"""
    if savepath is None:
        basedir = crops_path()
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
        if labels is None:
            continue
        depths, _ = load_labelled_videos(vid, getdepth=True, fillgaps=fillgaps)
        fr_num = frames.shape[0]
        for i in tqdm.tqdm(range(0, fr_num)):
            if labels[i] is not None:
                try:
                    fr_to_save = {}
                    frame = frames[i]
                    depth = depths[i]
                    frame, depth = transorm_rgd_depth(frame, depth, toofar=toofar, tooclose=tooclose)
                    frame = imresize(frame, resize_rate)
                    depth = depth_resize(depth, resize_rate)
                    label = labels[i][:, 0:2]
                    label *= [frame.shape[1], frame.shape[0]]
                    label = np.array(label, dtype=np.int32).tolist()
                    label = [[p[1], p[0]] for p in label]
                    frame = __add_padding(frame, frame.shape[1] - (frame.shape[1]//width_shrink_rate)*width_shrink_rate,
                                          frame.shape[0] - (frame.shape[0] // heigth_shrink_rate) * heigth_shrink_rate)
                    depth = __add_padding(depth, depth.shape[1] - (depth.shape[1]//width_shrink_rate)*width_shrink_rate,
                                          depth.shape[0] - (depth.shape[0] // heigth_shrink_rate) * heigth_shrink_rate)

                    depth = depth.squeeze()
                    depth = np.uint8(depth)
                    fr_to_save['frame'] = frame
                    coords = [__get_coord_from_labels(label)]
                    heat = u.get_heatmap_from_coords(frame, heigth_shrink_rate, width_shrink_rate,
                                                                      coords, overlapping_penalty)
                    fr_to_save['heatmap'] = __heatmap_to_uint8(heat)
                    fr_to_save['depth'] = depth
                    path = os.path.join(basedir, vid + "_" + str(i))
                    scio.savemat(path, fr_to_save)
                except ValueError as e:
                    print(vid + str(i) + " => " + e)


def create_dataset_shaded_heatmaps(videos_list=None, savepath=crops_path(), resize_rate=1.0, heigth_shrink_rate=10, width_shrink_rate=10,
                   overlapping_penalty=0.9, fillgaps=False, toofar=1500, tooclose=500, enlarge_heat=0.3,
                                   im_reg=reg.Regularizer(), he_r=reg.Regularizer()):
    """reads the videos specified as parameter and for each frame produces and saves a .mat file containing
    the frame, the corresponding heatmap indicating the position of the hand and the modified depth.
    :param tooclose: threshold value used to eliminate too close objects/values in the depth
    :param toofar: threshold value used to eliminate too far objects/values in the depth
    :param fillgaps: set to True to also get interpolated frames
    :param overlapping_penalty: penalty "inflicted" to the overlapping hands area in the images
    :param width_shrink_rate: shrink rate of heatmaps width wrt the resized image
    :param heigth_shrink_rate: shrink rate of heatmaps height wrt the resized image
    :param resize_rate: resize rate of the images (1 (default) is the original image)
    :param savepath: path of the folder where the produces .mat files will be saved. If left to the default value None,
    the /resources/hands_bounding_dataset/hands_rgbd_transformed folder will be used
    :param videos_list: list of videos you need the .mat files of. If left to the default value None, all videos will
    be exploited"""
    if savepath is None:
        basedir = crops_path()
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
        if labels is None:
            continue
        depths, _ = load_labelled_videos(vid, getdepth=True, fillgaps=fillgaps)
        fr_num = frames.shape[0]
        for i in tqdm.tqdm(range(0, fr_num)):
            if labels[i] is not None:
                try:
                    fr_to_save = {}
                    frame = frames[i]
                    depth = depths[i]
                    frame, depth = transorm_rgd_depth(frame, depth, toofar=toofar, tooclose=tooclose)
                    frame = imresize(frame, resize_rate)
                    depth = depth_resize(depth, resize_rate)
                    label = labels[i][:, 0:2]
                    label *= [frame.shape[1], frame.shape[0]]
                    label = np.array(label, dtype=np.int32).tolist()
                    label = [[p[1], p[0]] for p in label]
                    frame = __add_padding(frame, frame.shape[1] - (frame.shape[1]//width_shrink_rate)*width_shrink_rate,
                                          frame.shape[0] - (frame.shape[0] // heigth_shrink_rate) * heigth_shrink_rate)
                    depth = __add_padding(depth, depth.shape[1] - (depth.shape[1]//width_shrink_rate)*width_shrink_rate,
                                          depth.shape[0] - (depth.shape[0] // heigth_shrink_rate) * heigth_shrink_rate)

                    depth = depth.squeeze()
                    depth = np.uint8(depth)
                    frame = im_reg.apply(frame)
                    fr_to_save['frame'] = frame
                    coords = [__get_coord_from_labels(label)]
                    heat = u.get_heatmap_from_coords(frame, heigth_shrink_rate, width_shrink_rate,
                                                                      coords, overlapping_penalty)
                    coords = coords[0]
                    res_coords = [[l[0] // heigth_shrink_rate, l[1]//width_shrink_rate] for l in coords]
                    res_coords = __enlarge_coords(res_coords, enlarge_heat, np.shape(heat))
                    res_labels = [[l[0] // heigth_shrink_rate, l[1]//width_shrink_rate] for l in label]
                    heat = __shade_heatmap(heat, res_coords, res_labels)
                    heat = he_r.apply(heat)
                    heat = __heatmap_to_uint8(heat)
                    fr_to_save['heatmap'] = heat
                    depth = he_r.apply(depth)
                    fr_to_save['depth'] = depth
                    path = os.path.join(basedir, vid + "_" + str(i))
                    scio.savemat(path, fr_to_save)
                except ValueError as e:
                    print(vid + str(i) + " => " + e)


def __enlarge_coords(coord, enlarge, shape):
    image_height = shape[0]
    image_width = shape[1]
    up, down, left, right = __get_bounds(coord)
    up -= (down - up) * (enlarge / 2)
    down += (down - up) * (enlarge / 2)
    left -= (right - left) * (enlarge / 2)
    right += (right - left) * (enlarge / 2)
    up = int(up)
    down = int(down)
    left = int(left)
    right = int(right)
    if up < 0:
        up = 0
    if left < 0:
        left = 0
    if right >= image_width:
        right = image_width - 1
    if down >= image_height:
        down = image_height - 1
    return [[up, left], [up, right], [down, left], [down, right]]


def __get_bounds(coord):
    """given an array of 4 coordinates (x,y), simply computes and
    returns the highest and lowest vertical and horizontal points"""
    if len(coord) != 4:
        raise AttributeError("coord must be a set of 4 coordinates")
    x = [coord[i][0] for i in range(len(coord))]
    y = [coord[i][1] for i in range(len(coord))]
    up = np.min(x)
    down = np.max(x)
    left = np.min(y)
    right = np.max(y)
    return up, down, left, right


def __smoothen(heat, coords):
    h = [s[0] for s in coords]
    w = [w[1] for w in coords]
    min_h = np.min(h)
    max_h = np.max(h)
    min_w = np.min(w)
    max_w = np.max(w)
    h = max_h - min_h + 1
    w = max_w - min_w + 1
    copy = np.array(heat[min_h:max_h+1, min_w:max_w+1])
    ker_size = heat.shape[0] // h + heat.shape[1] // w
    ker = 2 * __ones_kernel(ker_size) / (ker_size * ker_size)
    heat = convolve(heat, ker, mode='same')
    heat[min_h:max_h + 1, min_w:max_w + 1] = copy
    return heat


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
                heat[i][j] = math.sqrt(1 - (heat[i][j] * heat[i][j])/(min_gauss * min_gauss))
            else:
                heat[i][j] = 1.

    return heat


def __sigmoid(x):
    return 1 / (1 + math.exp(-x))


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
    return abs(p[1] - m*p[0] - q)/math.sqrt(1 + m*m)


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
    return math.sqrt(math.pow(p1[0]-p2[0], 2) + math.pow(p1[1]-p2[1], 2))


@deprecated_fun(alternative=DatasetManager)
def read_dataset(path=crops_path(), verbosity=0, test_vids=None):
    """reads the .mat files present at the specified path. Note that those .mat files MUST be created using
    the create_dataset method
    :param verbosity: setting this parameter to True will make the method print the number of .mat files read
    every time it reads one
    :param path: path where the .mat files will be looked for. If left to its default value of None, the default path
    /resources/hands_bounding_dataset/hands_rgbd_transformed folder will be used
    :param test_vids: list of videos whose elements will be put in the test set. Note that is this parameter is not
    provided, only 3 arrays will be returned (frames, heatmaps, depths). If this is provided, 6 arrays are returned
    (frames, heatmaps, depths, test_frames, test_heatmaps, test_depths)
    """
    if path is None:
        basedir = crops_path()
    else:
        basedir = path
    samples = os.listdir(basedir)
    frames = []
    heatmaps = []
    depths = []
    t_frames = []
    t_heatmaps = []
    t_depths = []
    iterator = tqdm.tqdm(samples, file=sys.stdout, unit='frms') if verbosity == 1 else samples
    for name in iterator:
        realpath = os.path.join(basedir, name)
        matcontent = scio.loadmat(realpath)
        if test_vids is None or not __matches(name, test_vids):
            frames.append(matcontent['frame'])
            heatmaps.append(__heatmap_uint8_to_float32(matcontent['heatmap']))
            depths.append(matcontent['depth'])
        else:
            t_frames.append(matcontent['frame'])
            t_heatmaps.append(__heatmap_uint8_to_float32(matcontent['heatmap']))
            t_depths.append(matcontent['depth'])
    if test_vids is None:
        return frames, heatmaps, depths
    return frames, heatmaps, depths, t_frames, t_heatmaps, t_depths


@deprecated_fun(alternative=DatasetManager)
def read_dataset_random(path=crops_path(), number=1, verbosity=0, vid_list=None):
    """reads "number" different random .mat files present at the specified path. Note that those .mat files MUST be created using
    the create_dataset method
    :param verbosity: setting this parameter to True will make the method print the number of .mat files read
    every time it reads one
    :param path: path where the .mat files will be looked for. If left to its default value of None, the default path
    /resources/hands_bounding_dataset/hands_rgbd_transformed folder will be used
    :param number: number of elements to read
    :param vid_list: list of videos from which samples will be taken
    """
    if path is None:
        basedir = crops_path()
    else:
        basedir = path
    samples = os.listdir(basedir)
    if vid_list is not None:
        samples = [s for s in samples if __matches(s, vid_list)]
    tot = len(samples)
    if number > tot:
        raise ValueError("number must be smaller than the number of samples")
    random.shuffle(samples)
    samples = samples[:number]
    frames = []
    heatmaps = []
    depths = []
    iterator = tqdm.trange(number, file=sys.stdout, unit='frms') if verbosity == 1 else range(number)
    for idx in iterator:
        name = samples[idx]
        realpath = os.path.join(basedir, name)
        matcontent = scio.loadmat(realpath)
        frames.append(matcontent['frame'])
        heatmaps.append(__heatmap_uint8_to_float32(matcontent['heatmap']))
        depths.append(matcontent['depth'])
    return frames, heatmaps, depths


def __matches(s, leave_out):
    for stri in leave_out:
        if s.startswith(stri + "_"):
            return True
    return False


def shuffle_rgb_depth_heatmap(ri, di, hi):
    ri = np.array(ri)
    hi = np.array(hi)
    di = np.array(di)
    n = ri.shape[0]
    pos = list(range(0, n))
    n -= 0.01
    r1 = []
    d1 = []
    h1 = []
    while n > 0:
        rand = int(math.floor(random.uniform(0, n)))
        r1.append(ri[pos[rand]])
        d1.append(di[pos[rand]])
        h1.append(hi[pos[rand]])
        pos.pop(rand)
        n -= 1
    return np.array(r1), np.array(d1), np.array(h1)


def __get_coord_from_labels(lista):
    list_x = np.array([p[0] for p in lista])
    list_y = np.array([p[1] for p in lista])
    min_x = np.min(list_x)
    max_x = np.max(list_x)
    min_y = np.min(list_y)
    max_y = np.max(list_y)
    return [[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]


def load_all_frames_data(framespath, verbosity=0):
    framespathreal = resources_path(framespath)
    frames = cdc.read_frame_data(**cdc.default_read_rgb_args(framespathreal))
    depths = cdc.read_frame_data(**cdc.default_read_z16_args(framespathreal))
    frames = np.array(frames)
    depths = np.array(depths)
    if verbosity == 1:
        print("FRAMES SHAPE: ", frames.shape)
        print("LABELS SHAPE: ", depths.shape)
    return frames, depths


def get_numbered_frame(framespath, number, verbosity=0):
    frames1, depths1 = load_all_frames_data(framespath, verbosity)
    nframe = frames1[number]
    ndepth = depths1[number]
    return nframe, ndepth


def single_depth_frame_to_redblue(depthframe):
    depthsfirstframe = gtrbc.codec(np.array([depthframe]))
    return depthsfirstframe[0]


def show_frame(frame):
    u.showimage(frame)


def __right_derivative_kernel():
    return np.array([[-1, 1]])


def __left_derivative_kernel():
    return np.array([[1, -1]])


def __ones_kernel(dim):
    return np.ones([dim, dim])


def eliminate_too_far(depth, toofartheshold):
    depth[depth > toofartheshold] = 0
    return depth


def eliminate_too_close(depth, tooclosetheshold):
    depth[depth < tooclosetheshold] = 0
    return depth


def normalize_non_zeros(depth):
    depth[depth != 0] = 1
    return depth


def elementwise_product(frame, mapp):
    frame1 = np.multiply(frame[:, :, 0], mapp)
    frame2 = np.multiply(frame[:, :, 1], mapp)
    frame3 = np.multiply(frame[:, :, 2], mapp)
    frame_rec = np.dstack((frame1, frame2))
    frame_rec = np.dstack((frame_rec, frame3))
    return np.uint8(frame_rec)


def __heatmap_to_uint8(heat):
    heat = heat * 255
    heat.astype(np.uint8, copy=False)
    return heat


def __heatmap_uint8_to_float32(heat):
    heat.astype(np.float32, copy=False)
    heat = heat / 255
    return heat


def transorm_rgd_depth(frame, depth, showimages=False, toofar=1500, tooclose=500, toosmall=500):
    depth1 = np.squeeze(depth)
    if showimages:
        show_frame(depth1)
    depth1 = eliminate_too_far(depth1, toofar)
    depth1 = eliminate_too_close(depth1, tooclose)
    depth1 = normalize_non_zeros(depth1)
    if showimages:
        show_frame(depth1)
    # depth1 = convolve(depth1, __ones_kernel(3))
    # depth1 = convolve(depth1, __ones_kernel(3))
    # depth1 = convolve(depth1, __ones_kernel(3))
    # depth1 = normalize_non_zeros(depth1)
    # depth1 = eliminate_too_small_areas(depth1, toosmall)
    # frame1 = elementwise_product(frame, depth1)
    frame1 = frame
    if showimages:
        show_frame(frame1)
        show_frame(depth1)
    return frame1, depth1


def __add_padding(image, right_pad, bottom_pad):
    image = np.hstack((image, np.zeros([image.shape[0], right_pad, image.shape[2]], dtype=image.dtype)))
    image = np.vstack((image, np.zeros([bottom_pad, image.shape[1], image.shape[2]], dtype=image.dtype)))
    return image


def eliminate_too_small_areas(depth, toosmall=500):
    checked = np.zeros(depth.shape)
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            if depth[i][j] == 1 and checked[i][j] == 0:
                points = [[i, j]]
                checked[i][j] = 1
                for p in points:
                    if p[0] > 0 and checked[p[0] - 1][p[1]] == 0 and depth[p[0] - 1][p[1]] == 1:
                        points.append([p[0] - 1, p[1]])
                        checked[p[0] - 1][p[1]] = 1
                    if p[1] > 0 and checked[p[0]][p[1] - 1] == 0 and depth[p[0]][p[1] - 1] == 1:
                        points.append([p[0], p[1] - 1])
                        checked[p[0]][p[1] - 1] = 1
                    if p[1] < depth.shape[1] - 1 and checked[p[0]][p[1] + 1] == 0 and depth[p[0]][p[1] + 1] == 1:
                        points.append([p[0], p[1] + 1])
                        checked[p[0]][p[1] + 1] = 1
                    if p[0] < depth.shape[0] - 1 and checked[p[0] + 1][p[1]] == 0 and depth[p[0] + 1][p[1]] == 1:
                        points.append([p[0] + 1, p[1]])
                        checked[p[0] + 1][p[1]] = 1
                if len(points) < toosmall:
                    for p in points:
                        depth[p[0]][p[1]] = 0
    return depth


def timetest():
    firstframe, firstdepth = get_numbered_frame("rawcam/out-1520009971", 214)

    def realtest():
        firstframe1, firstdepth1 = transorm_rgd_depth(firstframe, firstdepth)
        return firstframe1, firstdepth1

    print(tm(realtest, number=1))


def save_some_frames_with_heatmap_and_crops(n):
    f, h, _ = read_dataset()
    path = resources_path("saves_for_report")
    os.makedirs(path, exist_ok=True)
    for i in range(n):
        os.makedirs(os.path.join(path, str(i)), exist_ok=True)
        pos = np.random.randint(0, len(f))
        f_ts = f[pos]
        h_ts = h[pos]
        f.pop(pos)
        h.pop(pos)
        u.save_image(os.path.join(path, str(i), "frame.jpg"), f_ts)
        heat_im = np.array(np.dstack((h_ts, h_ts, h_ts)) * 255, dtype=np.uint8)
        u.save_image(os.path.join(path, str(i), "heatmap.jpg"), heat_im)
        crop = u.get_crops_from_heatmap(image=f_ts, heatmap=h_ts, height_shrink_rate=4, width_shrink_rate=4)[0]
        u.save_image(os.path.join(path, str(i), "crop.jpg"), np.array(crop))


def create_sprite(n, num=0):
    f, h, _ = read_dataset(path=resources_path("datasets/dataset_for_report"))
    from data.datasets.crop.egohand_dataset_manager import read_dataset as r_ego
    f1, h1 = r_ego(path=resources_path("datasets/dataset_for_report2"))
    f = np.array(f)
    h = np.array(h)
    f1 = np.array(f1)
    h1 = np.array(h1)
    f = np.concatenate((f, f1))
    h = np.concatenate((h, h1))
    path = resources_path(os.path.join("saves_for_report", "sprite"))
    os.makedirs(path, exist_ok=True)
    sprite = None
    for i in range(n):
        row = None
        for j in range(n):
            pos = np.random.randint(0, len(f))
            f_ts = f[pos]
            h_ts = h[pos]
            heat3d = np.dstack((h_ts, h_ts, h_ts))
            heat3d[heat3d == 0] = 0.3
            if row is None:
                row = heat3d * f_ts
            else:
                row = np.array(np.hstack((row, heat3d * f_ts)), dtype=np.uint8)
        if sprite is None:
            sprite = row
        else:
            sprite = np.vstack((sprite, row))

    u.save_image(os.path.join(path, str(num) + ".jpg"), sprite)


if __name__ == '__main__':
    create_sprite(25)
