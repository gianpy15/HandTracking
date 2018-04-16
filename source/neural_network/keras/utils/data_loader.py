from hands_bounding_utils import hands_locator_from_rgbd as croputils
import hands_regularizer.regularizer as regularizer
from hands_regularizer.regularizer import Regularizer
from neural_network.keras.utils.naming import *
from junctions_locator_utils import junction_locator_ds_management as jlocutils
import numpy as np
import random as rnd
import sys

INDEPENDENT_FRAME_VIDEOS = ['egohands']


def load_dataset(train_samples, valid_samples, data_format=CROPPER,
                 random_dataset=False,
                 shuffle=True,
                 use_depth=False,
                 build_videos=None,
                 dataset_path=None,
                 verbose=False,
                 separate_valid=True,
                 independent_frames_ratio=0.3):
    merge_vids = False

    if data_format == CROPPER:
        read_dataset_random = croputils.read_dataset_random
        read_dataset = croputils.read_dataset
        create_dataset = create_crop_dataset
        need_expand_greys = True
        double_target = False
    else:
        read_dataset_random = jlocutils.read_dataset_random
        read_dataset = jlocutils.read_dataset
        create_dataset = create_joint_dataset
        need_expand_greys = False
        double_target = True

    if dataset_path is None:
        dataset_path = crops_path() if data_format == CROPPER else joints_path()

    if build_videos is not None:
        # if we want to build all videos we have the special token "ALL":
        if build_videos == "ALL":
            videos_list = None
        else:
            videos_list = build_videos
        if verbose:
            print("Building dataset on videos %s" % videos_list)
        create_dataset(videos_list=videos_list,
                       dataset_path=dataset_path)

    dataset_info = _get_available_dataset_stats(dataset_path)

    independent_frame_data = __load_indepdendent_videos(train_samples=int(train_samples * independent_frames_ratio),
                                                        valid_samples=int(valid_samples * independent_frames_ratio),
                                                        verbose=verbose,
                                                        random_read_f=read_dataset_random,
                                                        path=dataset_path,
                                                        dataset_info=dataset_info)

    dataset_info = __exclude_videos(dataset_info, INDEPENDENT_FRAME_VIDEOS)

    train_samples -= len(independent_frame_data['TRAIN'][0])
    valid_samples -= len(independent_frame_data['VALID'][0])

    train_vids, valid_vids = choose_train_valid_videos(dataset_info, train_samples, valid_samples)
    traincount = available_frames(dataset_info, train_vids)
    validcount = available_frames(dataset_info, valid_vids)

    if validcount < valid_samples or traincount < train_samples:
        availability = traincount + validcount
        requested = train_samples + valid_samples
        sys.stderr.write("WARNING: unable to load required dataset\n")
        sys.stderr.write("Cause: not enough data (%d available %d requested)\n" % (availability, requested))
        train_ratio = train_samples / requested
        train_samples = int(min(availability, requested) * train_ratio)
        train_vids, valid_vids = choose_train_valid_videos(dataset_info, train_samples, -1)
        if not separate_valid:
            merge_vids = True
            valid_samples = min(availability, requested) - train_samples
        else:
            train_samples = available_frames(dataset_info, train_vids)
            valid_samples = available_frames(dataset_info, valid_vids)

        sys.stderr.write("Using %d training samples and %d validation samples instead\n" % (train_samples,
                                                                                            valid_samples))
        sys.stderr.flush()

    if verbose:
        print("Chosen train videos: %s" % train_vids)
        print("Chosen valid videos: %s" % valid_vids)

    if random_dataset:
        if merge_vids:
            if verbose:
                print("Reading data...")
            imgs, maps, trd = read_dataset_random(path=dataset_path,
                                                  number=train_samples + valid_samples,
                                                  vid_list=train_vids + valid_vids,
                                                  verbosity=1 if verbose else 0)
            imgs, trd, maps = croputils.shuffle_rgb_depth_heatmap(imgs, trd, maps)
            train_imgs = imgs[:train_samples]
            train_maps = maps[:train_samples]
            train_trd = trd[:train_samples]
            valid_imgs = imgs[train_samples:train_samples + valid_samples]
            valid_maps = np.array(maps[train_samples:train_samples + valid_samples])
            valid_trd = trd[train_samples:train_samples + valid_samples]
        else:
            if verbose:
                print("Reading training data...")
            train_imgs, train_maps, train_trd = read_dataset_random(path=dataset_path,
                                                                    number=train_samples,
                                                                    vid_list=train_vids,
                                                                    verbosity=1 if verbose else 0)
            if verbose:
                print("Reading validation data...")
            valid_imgs, valid_maps, valid_trd = read_dataset_random(path=dataset_path,
                                                                    number=valid_samples,
                                                                    vid_list=valid_vids,
                                                                    verbosity=1 if verbose else 0)

    else:
        if verbose:
            print("Reading data...")
        if merge_vids:
            imgs, maps, trd = read_dataset(path=dataset_path,
                                           verbosity=1 if verbose else 0)
            imgs, trd, maps = croputils.shuffle_rgb_depth_heatmap(imgs, trd, maps)
            train_imgs = imgs[:train_samples]
            train_maps = maps[:train_samples]
            train_trd = trd[:train_samples]
            valid_imgs = imgs[train_samples:]
            valid_maps = maps[train_samples:]
            valid_trd = trd[train_samples:]
        else:
            train_imgs, train_maps, train_trd, \
            valid_imgs, valid_maps, valid_trd = read_dataset(path=dataset_path,
                                                             test_vids=valid_vids,
                                                             verbosity=1 if verbose else 0)

            train_imgs, train_maps, train_trd = train_imgs[0:train_samples], \
                                                train_maps[0:train_samples], \
                                                train_trd[0:train_samples]
            valid_imgs, valid_maps, valid_trd = valid_imgs[0:valid_samples], \
                                                valid_maps[0:valid_samples], \
                                                valid_trd[0:valid_samples]

    if len(independent_frame_data['TRAIN'][0]) != 0:
        train_imgs = np.concatenate((train_imgs, independent_frame_data['TRAIN'][0]))
        train_maps = np.concatenate((train_maps, independent_frame_data['TRAIN'][1]))
        train_trd = np.concatenate((train_trd, independent_frame_data['TRAIN'][2]))
    if len(independent_frame_data['VALID'][0]) != 0:
        valid_imgs = np.concatenate((valid_imgs, independent_frame_data['VALID'][0]))
        valid_maps = np.concatenate((valid_maps, independent_frame_data['VALID'][1]))
        valid_trd = np.concatenate((valid_trd, independent_frame_data['VALID'][2]))

    if shuffle:
        if verbose:
            print("Shuffling data...")
        train_imgs, train_trd, train_maps = croputils.shuffle_rgb_depth_heatmap(train_imgs, train_trd, train_maps)

    if need_expand_greys:
        if verbose:
            print("Formatting greys...")
        train_maps = np.expand_dims(train_maps, axis=np.ndim(train_maps))
        valid_maps = np.expand_dims(valid_maps, axis=np.ndim(valid_maps))
        if use_depth:
            train_trd = np.expand_dims(train_trd, axis=np.ndim(train_trd))
            valid_trd = np.expand_dims(valid_trd, axis=np.ndim(valid_trd))

    train_imgs = np.divide(train_imgs, 255.0, dtype=np.float32)
    valid_imgs = np.divide(valid_imgs, 255.0, dtype=np.float32)

    if use_depth:
        if verbose:
            print("Attaching depth...")
        train_input = np.concatenate((train_imgs, train_trd), axis=-1)
        valid_input = np.concatenate((valid_imgs, valid_trd), axis=-1)
    else:
        train_input = train_imgs
        valid_input = valid_imgs

    if verbose:
        print("Dataset ready!")

    if double_target:
        train_second_target = train_trd
        valid_second_target = valid_trd
    else:
        train_second_target = None
        valid_second_target = None

    ret = {TRAIN_IN: train_input,
           TRAIN_TARGET: train_maps,
           TRAIN_TARGET2: train_second_target,
           VALID_IN: valid_input,
           VALID_TARGET: valid_maps,
           VALID_TARGET2: valid_second_target}

    for k in ret.keys():
        if not isinstance(ret[k], np.ndarray):
            ret[k] = np.array(ret[k])

    return ret


def _get_available_dataset_stats(dataset_dir):
    framelist = os.listdir(dataset_dir)
    stats = {}
    for frame in framelist:
        name = frame.split(sep='_')[0]
        if name in stats.keys():
            stats[name] += 1
        else:
            stats[name] = 1
    return stats


def choose_train_valid_videos(dataset_info, train_samples, valid_samples=-1):
    available_vids = list(dataset_info.keys())
    rnd.shuffle(available_vids)

    def replenish_with_vids(samples):
        # option: take all remaining: ONLY USABLE FOR VALIDATION
        if samples < 0:
            return available_vids[:]

        vids = []
        left = samples
        while left > 0:
            if len(available_vids) == 0:
                # not enough data, this will be checked and handled later
                break
            chosen_vid = available_vids.pop()
            vids.append(chosen_vid)
            left -= dataset_info[chosen_vid]
        return vids

    return replenish_with_vids(train_samples), replenish_with_vids(valid_samples)


def available_frames(dataset_info, vidlist):
    count = 0
    availablevids = dataset_info.keys()
    for vid in vidlist:
        if vid in availablevids:
            count += dataset_info[vid]
    return count


def create_crop_dataset(videos_list, dataset_path):
    croputils.create_dataset_shaded_heatmaps(savepath=dataset_path, fillgaps=True,
                                             resize_rate=0.5, width_shrink_rate=4, heigth_shrink_rate=4,
                                             videos_list=videos_list)


def create_joint_dataset(videos_list, dataset_path):
    img_reg = regularizer.Regularizer()
    img_reg.fixresize(200, 200)
    hm_reg = regularizer.Regularizer()
    hm_reg.fixresize(100, 100)
    hm_reg.heatmaps_threshold(.5)
    jlocutils.create_dataset(savepath=dataset_path, fillgaps=True, im_regularizer=img_reg,
                             heat_regularizer=hm_reg, enlarge=.5, cross_radius=5,
                             videos_list=videos_list)


def __load_indepdendent_videos(train_samples, valid_samples, random_read_f, path, verbose=False, dataset_info=None):
    if dataset_info is not None:
        totframes = available_frames(dataset_info, INDEPENDENT_FRAME_VIDEOS)
        if train_samples + valid_samples > totframes:
            resize = totframes / (train_samples + valid_samples)
            train_samples = int(resize * train_samples)
            valid_samples = int(resize * valid_samples)
    imgs, maps, trd = random_read_f(path=path,
                                    number=train_samples + valid_samples,
                                    vid_list=INDEPENDENT_FRAME_VIDEOS,
                                    verbosity=1 if verbose else 0)
    return {'TRAIN': (imgs[:train_samples], maps[:train_samples], trd[:train_samples]),
            'VALID': (imgs[train_samples:], maps[train_samples:], trd[train_samples:])}


def __exclude_videos(dataset_info, videos):
    for vid in videos:
        if vid in dataset_info.keys():
            dataset_info[vid] = None
    return dataset_info

def load_joint_dataset(train_samples, valid_samples,
                       random_dataset=False,
                       shuffle=True,
                       build_videos=None,
                       dataset_path=None,
                       verbose=False,
                       separate_valid=True):
    return load_dataset(train_samples=train_samples,
                        valid_samples=valid_samples,
                        data_format=JLOCATOR,
                        random_dataset=random_dataset,
                        shuffle=shuffle,
                        use_depth=False,
                        build_videos=build_videos,
                        dataset_path=dataset_path,
                        verbose=verbose,
                        separate_valid=separate_valid)


def load_crop_dataset(train_samples, valid_samples,
                      random_dataset=False,
                      shuffle=True,
                      use_depth=False,
                      build_videos=None,
                      dataset_path=None,
                      verbose=False,
                      separate_valid=True):
    return load_dataset(train_samples=train_samples,
                        valid_samples=valid_samples,
                        data_format=CROPPER,
                        random_dataset=random_dataset,
                        shuffle=shuffle,
                        use_depth=use_depth,
                        build_videos=build_videos,
                        dataset_path=dataset_path,
                        verbose=verbose,
                        separate_valid=separate_valid)


if __name__ == '__main__':
    dataset = load_dataset(train_samples=10,
                           valid_samples=10,
                           random_dataset=True,
                           shuffle=True,
                           use_depth=False,
                           verbose=True)
    import matplotlib.pyplot as plot
    for img in dataset[TRAIN_IN]:
        plot.imshow(img)
        plot.show()

    for img in dataset[VALID_IN]:
        plot.imshow(img)
        plot.show()
