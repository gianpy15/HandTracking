import hands_bounding_utils.utils as u
import numpy as np
import source.hand_data_management.video_loader as vl
import hands_regularizer.regularizer as reg
import os
import data_manager.path_manager as pm
import tqdm
import scipy.io as scio
pm = pm.PathManager()


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
                   heat_regularizer=reg.Regularizer(), fillgaps=False, cross_radius=3, enlarge=0.2):
    """reads the videos specified as parameter and for each frame produces and saves a .mat file containing
    the frame, the corresponding heatmap indicating the position of the hand and the modified depth.
    :param fillgaps: set to True to also get interpolated frames
    :param im_regularizer: object used to regularize the images
    :param heat_regularizer: object used to regularize the heatmaps
    :param savepath: path of the folder where the produces .mat files will be saved. If left to the default value None,
    the /resources/hands_bounding_dataset/hands_rgbd_transformed folder will be used
    :param videos_list: list of videos you need the .mat files of. If left to the default value None, all videos will
    be exploited
    :param cross_radius: radius of the crosses of the heatmaps"""
    if savepath is None:
        basedir = pm.resources_path(os.path.join("hands_locator_dataset", "cuts_tranformed"))
    else:
        basedir = savepath
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    framesdir = pm.resources_path("framedata")
    if videos_list is None:
        vids = os.listdir(framesdir)
        vids = [x for x in vids if os.path.isdir(os.path.join(framesdir, x))]
    else:
        vids = videos_list
    for vid in tqdm.tqdm(vids):
        frames, labels = load_labelled_videos(vid, fillgaps=fillgaps)
        # depths, _ = load_labelled_videos(vid, getdepth=True, fillgaps=fillgaps)
        fr_num = frames.shape[0]
        for i in tqdm.tqdm(range(0, fr_num)):
            if labels[i] is not None:
                try:
                    fr_to_save = {}
                    frame = frames[i]
                    label = labels[i][:, 0:2]
                    visible = labels[i][:, 2:3]
                    label *= [frame.shape[1], frame.shape[0]]
                    label = np.array(label, dtype=np.int32).tolist()
                    label = [[p[1], p[0]] for p in label]
                    coords = __get_coord_from_labels(label)
                    cut = u.crop_from_coords(frame, coords, enlarge)
                    heatmaps = __create_21_heatmaps(label, coords,
                                                    np.shape(frame), cross_radius, enlarge)
                    cut = im_regularizer.apply(cut)
                    heatmaps = heat_regularizer.apply_on_batch(heatmaps)
                    heatmaps = __heatmaps_dim_reducer(heatmaps)
                    heatmaps = __stack_heatmaps(heatmaps)
                    path = os.path.join(basedir, vid + str(i))
                    __persist_frame(path, cut, heatmaps, visible)
                except ValueError as e:
                    print("Error " + e + " on vid " + vid + str(i))


def __heatmaps_dim_reducer(heatmaps):
    heatris = []
    for he in heatmaps:
        heat = he[:, :, 0:1]
        heatris.append(heat)
    return heatris


def read_dataset(path=None, verbosity=0):
    """reads the .mat files present at the specified path. Note that those .mat files MUST be created using
    the create_dataset method
    :param verbosity: setting this parameter to True will make the method print the number of .mat files read
    every time it reads one
    :param path: path where the .mat files will be looked for. If left to its default value of None, the default path
    /resources/hands_bounding_dataset/hands_rgbd_transformed folder will be used
    """
    if path is None:
        basedir = pm.resources_path(os.path.join("hands_locator_dataset", "cuts_tranformed"))
    else:
        basedir = path
    samples = os.listdir(basedir)
    i = 0
    tot = len(samples)
    cuts = []
    heatmaps = []
    visible = []
    for name in samples:
        if verbosity == 1:
            print("Reading image: ", i, " of ", tot)
            i += 1
        realpath = os.path.join(basedir, name)
        readcuts, readheats, readvis = __read_frame(realpath)
        cuts.append(readcuts)
        heatmaps.append(readheats)
        visible.append(readvis)
    return cuts, heatmaps, visible


def __create_21_heatmaps(label, coords_for_cut, original_shape, cross_radius, enlarge):
    heatmaps = []
    for l in label:
        heat = np.zeros(original_shape)
        heat = __set_cross(heat, l, cross_radius)
        heat = u.crop_from_coords(heat, coords_for_cut, enlarge)
        heatmaps.append(heat)
    return np.array(heatmaps)


def __stack_heatmaps(heatmaps):
    stacked_heats = np.dstack(tuple(heatmaps))
    return np.array(stacked_heats)


def __set_cross(heatmap, center, radius):
    for i in range(-radius, radius):
        for j in range(-radius, radius):
            if abs(i) + abs(j) <= radius and 0 <= center[0] + i <= heatmap.shape[0] \
                    and 0 <= center[1] + j <= heatmap.shape[1]:
                heatmap[center[0]+i][center[1]+j] = [1, 1, 1]
    return heatmap


def __get_coord_from_labels(lista):
    list_x = np.array([p[0] for p in lista])
    list_y = np.array([p[1] for p in lista])
    min_x = np.min(list_x)
    max_x = np.max(list_x)
    min_y = np.min(list_y)
    max_y = np.max(list_y)
    return [[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]


def __persist_frame(path, cut, heatmaps, visible_flags):
    fr_to_save = {'cut': cut,
                  'heatmap_array': np.array(heatmaps * 255, dtype=np.uint8),
                  'visible': np.array(visible_flags, dtype=np.float32)}
    scio.savemat(path, fr_to_save)


def __read_frame(path):
    matcontent = scio.loadmat(path)
    return matcontent['cut'], matcontent['heatmap_array'] / 255.0, matcontent['visible']


if __name__ == '__main__':
    im_r = reg.Regularizer()
    im_r.fixresize(200, 200)
    im_r.percresize(0.5)
    h_r = reg.Regularizer()
    h_r.fixresize(200, 200)
    h_r.percresize(0.5)
    h_r.heatmaps_threshold(0.5)
    create_dataset(["handsGianpy"], im_regularizer=im_r, heat_regularizer=h_r, enlarge=0.5, cross_radius=5)
    c, h, v = read_dataset()
    u.showimage(c[1])
    print(np.shape(h))
    # show heatmap for the first junction of the second item
    u.showimage(u.heatmap_to_rgb(h[1][:, :, 0:1]))
    print(v[1])
