from data.datasets.crop import hands_locator_from_rgbd as croputils
from data import *
from library import *
from data.datasets.jlocator import junction_locator_ds_management as jlocutils
from data.datasets.crop import egohand_dataset_manager as egoutils
import numpy as np
import re

# List of video first names not including depth
NO_DEPTH_VIDEOS = ['CARDS', 'CHESS', 'JENGA', 'PUZZLE']

# Defines to organize components during reading
OUTPUT_STD_FORMAT_LEN = 4
IMG_ORD = 0
TARG1_ORD = 1
TARG2_ORD = 2
DEPTH_ORD = 3


def unavailable_functionality(*args, **kwargs):
    raise NotImplementedError("The requested functionality is not available")


# Reading function mapping
READ_FUNCTIONS = {
    CROPPER: {
        RAND: {
            'DEPTH': (croputils.read_dataset_random, (IMG_ORD, TARG1_ORD, DEPTH_ORD)),
            'NODEPTH': (egoutils.read_dataset_random, (IMG_ORD, TARG1_ORD))
        },
        SEQUENTIAL: {
            'DEPTH': (croputils.read_dataset, (IMG_ORD, TARG1_ORD, DEPTH_ORD)),
            'NODEPTH': (egoutils.read_dataset, (IMG_ORD, TARG1_ORD))
        }
    },
    JLOCATOR: {
        RAND: {
            'DEPTH': (unavailable_functionality, (IMG_ORD, TARG1_ORD, TARG2_ORD, DEPTH_ORD)),
            'NODEPTH': (jlocutils.read_dataset_random, (IMG_ORD, TARG1_ORD, TARG2_ORD))
        },
        SEQUENTIAL: {
            'DEPTH': (unavailable_functionality, (IMG_ORD, TARG1_ORD, TARG2_ORD, DEPTH_ORD)),
            'NODEPTH': (jlocutils.read_dataset, (IMG_ORD, TARG1_ORD, TARG2_ORD))
        }
    }
}


# READING_FUNCTION dictionary wrapper
def read_function(data_type, mode, depth):
    if mode == SEQUENTIAL:
        return unavailable_functionality
    read_f, swaps = READ_FUNCTIONS[data_type][mode][depth]
    return lambda path, vid_list, number: swap_elements(read_f(path=path,
                                                               vid_list=vid_list,
                                                               number=number),
                                                        swaps=swaps)


def swap_elements(elem, swaps):
    out = [None] * OUTPUT_STD_FORMAT_LEN
    for idx in range(len(swaps)):
        out[swaps[idx]] = elem[idx]
    return out


def load_dataset(train_samples, valid_samples, data_format=CROPPER,
                 use_depth=False,
                 dataset_path=None,
                 exclude=None):
    if exclude is None:
        exclude = []

    if use_depth:
        exclude += NO_DEPTH_VIDEOS

    if dataset_path is None:
        dataset_path = crops_path() if data_format == CROPPER else joints_path()

    dataset_info = __exclude_videos(_get_available_dataset_stats(dataset_path), exclude)

    if len(dataset_info.keys()) == 0:
        log("No eligible videos have been found.", level=ERRORS)

    train_vids, valid_vids, shared_vids = choose_train_valid_shared_videos(dataset_info, train_samples, valid_samples)

    log("Loading training data...", level=COMMENTARY)
    train_reads = min(train_samples, available_frames(dataset_info, train_vids))
    train_dataset = __load_samples(videos_list=train_vids,
                                   path=dataset_path,
                                   number=train_reads,
                                   use_depth=use_depth,
                                   in_sequence=False,
                                   data_type=data_format)

    log("Loading validation data...", level=COMMENTARY)
    valid_reads = min(valid_samples, available_frames(dataset_info, valid_vids))
    valid_dataset = __load_samples(videos_list=valid_vids,
                                   path=dataset_path,
                                   number=valid_reads,
                                   use_depth=use_depth,
                                   in_sequence=False,
                                   data_type=data_format)

    if len(shared_vids) > 0:
        train_missing = max(train_samples - train_reads, 0)
        valid_missing = max(valid_samples - valid_reads, 0)
        reading = min(train_missing + valid_missing, available_frames(dataset_info, shared_vids))
        log("WARNING: Not enough eligible videos to balance train and validation separately", level=WARNINGS)
        log("One video will be shared to grant balance", level=WARNINGS)
        log("Loading shared data...", level=COMMENTARY)
        shared_dataset = __load_samples(videos_list=shared_vids,
                                        path=dataset_path,
                                        number=reading,
                                        use_depth=use_depth,
                                        in_sequence=False,
                                        data_type=data_format)

        if reading < train_missing + valid_missing:
            train_missing = int(round(train_missing * reading / (train_missing + valid_missing)))
            valid_missing = reading - train_missing
        idx1 = train_missing
        idx2 = train_missing + valid_missing
        # Attach available shared data to current datasets
        for (ds, reads, start, stop) in ((train_dataset, train_reads, 0, idx1),
                                         (valid_dataset, valid_reads, idx1, idx2)):
            for comp in (GENERIC_IN, GENERIC_TARGET, GENERIC_TARGET2):
                if reads > 0:
                    ds[comp] = np.concatenate((ds[comp],
                                               shared_dataset[comp][start:stop]))
                else:
                    ds[comp] = shared_dataset[comp][start:stop]
        train_reads += train_missing
        valid_reads += valid_missing

    if train_reads < train_samples or valid_reads < valid_samples:
        log("WARNING: Unable to load the requested number of frames", level=IMPORTANT_WARNINGS)
        log("Loaded train samples: %d / %d" % (train_reads, train_samples), level=IMPORTANT_WARNINGS)
        log("Loaded valid samples: %d / %d" % (valid_reads, valid_samples), level=IMPORTANT_WARNINGS)

    log("Dataset ready!", level=COMMENTARY)

    ret = {TRAIN_IN: train_dataset[GENERIC_IN],
           TRAIN_TARGET: train_dataset[GENERIC_TARGET],
           TRAIN_TARGET2: train_dataset[GENERIC_TARGET2],
           VALID_IN: valid_dataset[GENERIC_IN],
           VALID_TARGET: valid_dataset[GENERIC_TARGET],
           VALID_TARGET2: valid_dataset[GENERIC_TARGET2]}

    for k in ret.keys():
        if not isinstance(ret[k], np.ndarray):
            ret[k] = np.array(ret[k])

    return ret


# #################################### LOADING UTILITIES ##########################################


def __load_samples(videos_list, path, number=1, use_depth=False, in_sequence=False, data_type=CROPPER):
    depthtoken = 'DEPTH' if use_depth else 'NODEPTH'
    mode = SEQUENTIAL if in_sequence else RAND
    dtype = data_type
    log("Loading data from %s" % videos_list, level=COMMENTARY)
    out = read_function(data_type=dtype,
                        mode=mode,
                        depth=depthtoken)(path=path,
                                          vid_list=videos_list,
                                          number=number)
    if data_type is CROPPER:
        # need to expand the grayscales
        for idx in (DEPTH_ORD, TARG1_ORD):
            if out[idx] is not None:
                out[idx] = np.expand_dims(out[idx], axis=np.ndim(out[idx]))

    if out[DEPTH_ORD] is not None:
        out[IMG_ORD] = np.concatenate((out[IMG_ORD], out[DEPTH_ORD]), axis=-1)
        out[DEPTH_ORD] = None

    out[IMG_ORD] = np.divide(out[IMG_ORD], 255.0, dtype=np.float32)

    for idx in range(len(out)):
        if out[idx] is None:
            out[idx] = []

    return {GENERIC_IN: out[IMG_ORD],
            GENERIC_TARGET: out[TARG1_ORD],
            GENERIC_TARGET2: out[TARG2_ORD]}


# ########################### DATASET STATS #####################################


def __exclude_videos(dataset_info, videos):
    to_be_deleted = [vid for vid in dataset_info.keys() if
                     any([re.match(vidreg, vid) for vidreg in videos])]
    for vid in to_be_deleted:
        del dataset_info[vid]
    return dataset_info


def _get_available_dataset_stats(dataset_dir):
    framelist = os.listdir(dataset_dir)
    stats = {}
    for frame in framelist:
        name_match = re.match("(?P<vid_name>^.*)_[^_]*\.mat$", frame)
        if name_match is not None:
            name = name_match.groups("vid_name")[0]
        else:
            continue
        if name in stats.keys():
            stats[name] += 1
        else:
            stats[name] = 1
    return stats


def available_frames(dataset_info, vidlist):
    count = 0
    availablevids = dataset_info.keys()
    for vid in vidlist:
        if vid in availablevids:
            count += dataset_info[vid]
    return count


# ################################### TRAIN-VALID SEPARATION ################################


def choose_train_valid_shared_videos(dataset_info, train, valid):
    assigned_to_train = 0
    assigned_to_valid = 0
    available_videos = list(dataset_info.keys())
    available_videos.sort(key=dataset_info.__getitem__)
    trainvids = []
    validvids = []
    train_satis = 0. if train > 0 else np.inf
    valid_satis = 0. if valid > 0 else np.inf

    def count_smaller_than(value):
        c = 0
        for v in available_videos:
            if dataset_info[v] < value:
                c += 1
        return c

    while len(available_videos) > 0:
        nextbatch, missing = (trainvids, train - assigned_to_train) if train_satis <= valid_satis \
            else (validvids, valid - assigned_to_valid)
        if missing <= 0 or missing >= dataset_info[available_videos[-1]]:
            vid = available_videos.pop()
        else:
            vid = available_videos.pop(count_smaller_than(missing))
        nextbatch.append(vid)
        if nextbatch is trainvids:
            assigned_to_train += dataset_info[vid]
            train_satis = assigned_to_train / train if train > 0 else np.inf
        else:
            assigned_to_valid += dataset_info[vid]
            valid_satis = assigned_to_valid / valid if valid > 0 else np.inf

    if train_satis >= 1 and valid_satis >= 1:
        return trainvids, validvids, []
    elif train_satis < 1 and valid_satis < 1:
        return trainvids, validvids, []
    else:
        poolbatch, target = (trainvids, valid - assigned_to_valid) if train_satis >= valid_satis \
            else (validvids, train - assigned_to_train)
        poolbatch.sort(key=dataset_info.__getitem__)
        shared = poolbatch.pop(count_smaller_than(target))
        return trainvids, validvids, [shared]


# ################################## WRAPPERS ###############################################


def load_joint_dataset(train_samples, valid_samples,
                       dataset_path=None,
                       exclude=None):
    return load_dataset(train_samples=train_samples,
                        valid_samples=valid_samples,
                        data_format=JLOCATOR,
                        use_depth=False,
                        dataset_path=dataset_path,
                        exclude=exclude)


def load_crop_dataset(train_samples, valid_samples,
                      use_depth=False,
                      dataset_path=None,
                      exclude=None):
    return load_dataset(train_samples=train_samples,
                        valid_samples=valid_samples,
                        data_format=CROPPER,
                        use_depth=use_depth,
                        dataset_path=dataset_path,
                        exclude=exclude)


if __name__ == '__main__':
    set_verbosity(DEBUG)
    import sys
    set_log_file(sys.stderr)
    ds = load_crop_dataset(train_samples=11,
                           valid_samples=10,
                           exclude=['handsD'])
    import matplotlib.pyplot as pplot

    for img in np.concatenate((ds[TRAIN_IN], ds[VALID_IN])):
        pplot.imshow(img)
        pplot.show()
