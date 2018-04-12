from hands_bounding_utils.hands_locator_from_rgbd import *
from data_manager.path_manager import resources_path
from hands_regularizer.regularizer import Regularizer
import random as rnd
import sys

DEFAULT_DATASET_PATH = resources_path("hands_bounding_dataset", "network_test")

TRAIN_IN = 'TRAIN_IN'
TRAIN_TARGET = 'TRAIN_TARGET'
VALID_IN = 'TEST_IN'
VALID_TARGET = 'TEST_TARGET'


def load_dataset(train_samples, valid_samples,
                 random_dataset=False,
                 shuffle=True,
                 use_depth=False,
                 build_videos=None,
                 dataset_path=None,
                 verbose=False):
    if dataset_path is None:
        dataset_path = DEFAULT_DATASET_PATH

    if verbose:
        print("Dataset location: %s" % dataset_path)

    if build_videos is not None:
        # if we want to build all videos we have the special token "ALL":
        if build_videos == "ALL":
            videos_list = None
        else:
            videos_list = build_videos
        if verbose:
            print("Building dataset on videos %s" % videos_list)
        create_dataset(savepath=dataset_path, fillgaps=True,
                       resize_rate=0.5, width_shrink_rate=4, heigth_shrink_rate=4,
                       videos_list=videos_list)

    dataset_info = _get_available_dataset_stats(dataset_path)

    train_vids, valid_vids = choose_train_valid_videos(dataset_info, train_samples, valid_samples)

    traincount = available_frames(dataset_info, train_vids)
    validcount = available_frames(dataset_info, valid_vids)

    if validcount < valid_samples or traincount < train_samples:
        availability = traincount + validcount
        requested = train_samples + valid_samples
        sys.stderr.write("WARNING: unable to load required dataset\n")
        sys.stderr.write("Cause: not enough data (%d available %d requested)\n" % (availability, requested))
        train_ratio = train_samples / requested
        train_samples = int(availability * train_ratio)
        train_vids, valid_vids = choose_train_valid_videos(dataset_info, train_samples, -1)
        train_samples = available_frames(dataset_info, train_vids)
        valid_samples = available_frames(dataset_info, valid_vids)
        sys.stderr.write("Using %d training samples and %d validation samples instead\n" % (train_samples,
                                                                                            valid_samples))
        sys.stderr.flush()

    if verbose:
        print("Chosen train videos: %s" % train_vids)
        print("Chosen valid videos: %s" % valid_vids)

    if random_dataset:
        if verbose:
            print("Reading training data...")
        train_imgs, train_maps, train_depths = read_dataset_random(path=dataset_path,
                                                                   number=train_samples,
                                                                   leave_out=valid_vids,
                                                                   verbosity=1 if verbose else 0)
        if verbose:
            print("Reading validation data...")
        valid_imgs, valid_maps, valid_depths = read_dataset_random(path=dataset_path,
                                                                   number=valid_samples,
                                                                   leave_out=train_vids,
                                                                   verbosity=1 if verbose else 0)

    else:
        if verbose:
            print("Reading data...")
        train_imgs, train_maps, train_depths, \
        valid_imgs, valid_maps, valid_depths = read_dataset(path=dataset_path,
                                                            leave_out=valid_vids,
                                                            verbosity=1 if verbose else 0)

        train_imgs, train_maps, train_depths = train_imgs[0:train_samples], \
                                               train_maps[0:train_samples], \
                                               train_depths[0:train_samples]
        valid_imgs, valid_maps, valid_depths = valid_imgs[0:valid_samples], \
                                               valid_maps[0:valid_samples], \
                                               valid_depths[0:valid_samples]

    if shuffle:
        if verbose:
            print("Shuffling data...")
        train_imgs, train_depths, train_maps = shuffle_rgb_depth_heatmap(train_imgs, train_depths, train_maps)

    if verbose:
        print("Formatting greys...")
    train_maps = np.expand_dims(train_maps, axis=np.ndim(train_maps))
    valid_maps = np.expand_dims(valid_maps, axis=np.ndim(valid_maps))
    if use_depth:
        train_depths = np.expand_dims(train_depths, axis=np.ndim(train_depths))
        valid_depths = np.expand_dims(valid_depths, axis=np.ndim(valid_depths))

    reg = Regularizer()
    reg.normalize()

    if verbose:
        print("Normalizing images...")
    train_imgs = reg.apply_on_batch(np.array(train_imgs))
    valid_imgs = reg.apply_on_batch(np.array(valid_imgs))

    if use_depth:
        if verbose:
            print("Attaching depth...")
        train_input = np.concatenate((train_imgs, train_depths), axis=-1)
        valid_input = np.concatenate((valid_imgs, valid_depths), axis=-1)
    else:
        train_input = train_imgs
        valid_input = valid_imgs

    if verbose:
        print("Dataset ready!")

    return {TRAIN_IN: train_input,
            TRAIN_TARGET: train_maps,
            VALID_IN: valid_input,
            VALID_TARGET: valid_maps}


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
    for vid in vidlist:
        count += dataset_info[vid]
    return count
