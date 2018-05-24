import json
import numpy as np
import tqdm
import math
from scipy import io as scio
from scipy.misc import imresize
import data.datasets.crop.utils as u
from data.naming import *
import data.regularization.regularizer as reg
from data.datasets.reading.dataset_manager import DatasetManager


def create_dataset_shaded_heatmaps(dspath=None, savepath=jsonhands_path(), heigth_shrink_rate=4, width_shrink_rate=4,
                                enlarge_heat=0.3, im_reg=reg.Regularizer(), he_r=reg.Regularizer(),
                                   resize_rate=1.0):
    if dspath is None:
        dspath = resources_path("jsonHands")
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    dspath = resources_path(dspath)
    labels_dir = os.path.join(dspath, "labels")
    framesdir = os.path.join(dspath, "images")
    frameslist = os.listdir(framesdir)
    final_base_path = resources_path(savepath)
    done_dict = dict()
    for frame in tqdm.tqdm(frameslist):
        fr_to_save = {}
        f_pruned_name = __remove_lr(frame)
        try:
            _ = done_dict[f_pruned_name]
            continue
        except KeyError:
            done_dict[f_pruned_name] = 1

        frame_l = __read_frame(os.path.join(framesdir, f_pruned_name + "_l.jpg"))
        frame_r = __read_frame(os.path.join(framesdir, f_pruned_name + "_r.jpg"))
        label_l = __read_label(os.path.join(labels_dir, f_pruned_name + "_l.json"))
        label_r = __read_label(os.path.join(labels_dir, f_pruned_name + "_r.json"))

        if frame_l is None:
            res_h = frame_r.shape[0]/480
            res_w = frame_r.shape[1]/640
        else:
            res_h = frame_l.shape[0] / 480
            res_w = frame_l.shape[1] / 640
        if frame_l is not None:
            frame_l = imresize(frame_l, (480, 640))
        elif frame_r is not None:
            frame_l = imresize(frame_r, (480, 640))
        else:
            continue
        frame_l = imresize(frame_l, resize_rate)
        label_l = [[p[1] * resize_rate / res_h, p[0] * resize_rate / res_w] for p in label_l]
        label_r = [[p[1] * resize_rate / res_h, p[0] * resize_rate / res_w] for p in label_r]

        frame_l = __add_padding(frame_l, frame_l.shape[1] - (frame_l.shape[1]//width_shrink_rate)*width_shrink_rate,
                                frame_l.shape[0] - (frame_l.shape[0] // heigth_shrink_rate) * heigth_shrink_rate)

        frame_l = im_reg.apply(frame_l)
        fr_to_save['frame'] = frame_l
        coords_r = [__get_coord_from_labels(label_r)]
        coords_l = [__get_coord_from_labels(label_l)]
        heat1 = np.zeros([frame_l.shape[0]//heigth_shrink_rate, frame_l.shape[1]//width_shrink_rate])
        heat2 = np.zeros([frame_l.shape[0]//heigth_shrink_rate, frame_l.shape[1]//width_shrink_rate])
        coords_r = coords_r[0]
        coords_l = coords_l[0]
        res_coords_r = [[l[0] // heigth_shrink_rate, l[1] // width_shrink_rate] for l in coords_r]
        res_coords_r = __enlarge_coords(res_coords_r, enlarge_heat, np.shape(heat2))
        res_coords_l = [[l[0] // heigth_shrink_rate, l[1] // width_shrink_rate] for l in coords_l]
        res_coords_l = __enlarge_coords(res_coords_l, enlarge_heat, np.shape(heat1))

        res_labels_l = [[l[0] // heigth_shrink_rate, l[1] // width_shrink_rate] for l in label_l]
        res_labels_r = [[l[0] // heigth_shrink_rate, l[1] // width_shrink_rate] for l in label_r]
        heat1 = __shade_heatmap(heat1, res_coords_l, res_labels_l)
        heat2 = __shade_heatmap(heat2, res_coords_r, res_labels_r)
        heat = heat1 + heat2
        heat[heat > 1] = 1
        heat = he_r.apply(heat)
        heat = __heatmap_to_uint8(heat)
        fr_to_save['heatmap'] = heat
        path = os.path.join(final_base_path, f_pruned_name + ".mat")
        scio.savemat(path, fr_to_save)


def __read_frame(path):
    try:
        frame = u.read_image(path)
        return frame
    except Exception:
        return None


def __read_label(path):
    try:
        json_data = open(path)
        data = json.load(json_data)
        return data['hand_pts']
    except Exception:
        return []

def __remove_lr(name):
    split = name.split("_")
    ris = ""
    tot = len(split) - 1
    for i in range(tot):
        ris += split[i]
        if i != tot - 1:
            ris += "_"
    return ris


def __enlarge_coords(coord, enlarge, shape):
    if not coord:
        return []
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


def __shade_heatmap(heat, square_coords, joint_coords):
    if not square_coords or not joint_coords:
        return heat
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


def __get_coord_from_labels(lista):
    if not lista:
        return []
    list_x = np.array([p[0] for p in lista])
    list_y = np.array([p[1] for p in lista])
    min_x = np.min(list_x)
    max_x = np.max(list_x)
    min_y = np.min(list_y)
    max_y = np.max(list_y)
    return [[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]


def show_frame(frame):
    u.showimage(frame)


def __heatmap_to_uint8(heat):
    heat = heat * 255
    heat.astype(np.uint8, copy=False)
    return heat


def __heatmap_uint8_to_float32(heat):
    heat.astype(np.float32, copy=False)
    heat = heat / 255
    return heat


def __add_padding(image, right_pad, bottom_pad):
    image = np.hstack((image, np.zeros([image.shape[0], right_pad, image.shape[2]], dtype=image.dtype)))
    image = np.vstack((image, np.zeros([bottom_pad, image.shape[1], image.shape[2]], dtype=image.dtype)))
    return image


if __name__ == '__main__':
    create_dataset_shaded_heatmaps(resize_rate=0.5)