import source.hand_data_management.camera_data_conversion as cdc
import source.hand_data_management.video_loader as vl
import source.hand_data_management.grey_to_redblue_codec as gtrbc
import numpy as np
import tqdm
import os
import math
from scipy import io as scio
from scipy.ndimage import convolve
from scipy.misc import imresize
import hands_bounding_utils.utils as u
from data_manager.path_manager import PathManager
from timeit import timeit as tm

pm = PathManager()


def load_labelled_videos(vname, getdepth=False, fillgaps=False, gapflags=False, verbosity=0):
    frames, labels = vl.load_labeled_video(vname, getdepth, fillgaps, gapflags)
    frames = np.array(frames)
    labels = np.array(labels)
    if verbosity == 1:
        print("FRAMES SHAPE: ", frames.shape)
        print("LABELS SHAPE: ", labels.shape)
    return frames, labels


def depth_resize(depth, rr):
    depth = depth.reshape([depth.shape[0], depth.shape[1], 1])
    depth = np.dstack((depth, depth, depth))
    return imresize(depth, rr)[:, :, 0:1]


def create_dataset(videos_list=None, savepath=None, resize_rate=1.0, heigth_shrink_rate=10, width_shrink_rate=10,
                   overlapping_penalty=0.9, fillgaps=False, toofar=1500, tooclose=500):
    if savepath is None:
        basedir = pm.resources_path(os.path.join("hands_bounding_dataset", "hands_rgbd_tranformed"))
    else:
        basedir = savepath
    framesdir = pm.resources_path("framedata")
    if videos_list is None:
        vids = os.listdir(framesdir)
        vids.remove("contributors.txt")
    else:
        vids = videos_list
    for vid in tqdm.tqdm(vids):
        frames, labels = load_labelled_videos(vid, fillgaps=fillgaps)
        depths, _ = load_labelled_videos(vid, getdepth=True, fillgaps=fillgaps)
        fr_num = frames.shape[0]
        for i in range(0, fr_num):
            if labels[i] is not None:
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
                fr_to_save['heatmap'] = u.get_heatmap_from_coords(frame, heigth_shrink_rate, width_shrink_rate,
                                                                  coords, overlapping_penalty)
                fr_to_save['depth'] = depth
                path = os.path.join(basedir, vid + str(i))
                scio.savemat(path, fr_to_save)


def read_dataset(path=None, verbosity=0):
    if path is None:
        basedir = pm.resources_path(os.path.join("hands_bounding_dataset", "hands_rgbd_tranformed"))
    else:
        basedir = path
    samples = os.listdir(basedir)
    i = 0
    tot = len(samples)
    frames = []
    heatmaps = []
    depths = []
    for name in samples:
        if verbosity == 1:
            print("Reading image: ", i, " of ", tot)
            i += 1
        realpath = os.path.join(basedir, name)
        matcontent = scio.loadmat(realpath)
        frames.append(matcontent['frame'])
        heatmaps.append(matcontent['heatmap'])
        depths.append(matcontent['depth'])
    return frames, heatmaps, depths


def __get_coord_from_labels(lista):
    list_x = np.array([p[0] for p in lista])
    list_y = np.array([p[1] for p in lista])
    min_x = np.min(list_x)
    max_x = np.max(list_x)
    min_y = np.min(list_y)
    max_y = np.max(list_y)
    return [[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]


def load_all_frames_data(framespath, verbosity=0):
    framespathreal = pm.resources_path(framespath)
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


if __name__ == '__main__':
    # firstframe, firstdepth = get_numbered_frame("rawcam/out-1520009971", 214)

    # firstframe1, firstdepth1 = transorm_rgd_depth(firstframe, firstdepth, showimages=True)

    # timetest()
    create_dataset()
    f, h, d = read_dataset()
    u.showimage(f[1])
    u.showimage(h[1])
    u.showimage(d[1])
    u.showimages(u.get_crops_from_heatmap(f[1], h[1]))
