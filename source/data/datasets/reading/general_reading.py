import scipy.io as scio
from data import *
import os
import numpy as np
from data.datasets.reading.exceptions import SkipFrameException

# This module includes functions for generic formatted reading of mat files
# All functions are synchrnonous and I/O intensive.
# formats are specified by a sequence of (key, consumer) pairs
#   - key: specifies the mat field to access for each file
#   - consumer: a function to digest the mat content to produce preprocessed data
#           (example: directly normalized, or dequantized data)
# formatdicts are dictionaries of formats
#
# See data.datasets.reading.formatting.py for more info about format specifications


def _read_frame(path: str, *, format):
    """
    Read a single specified frame applying the specified format.
    :param path: the path of the frame to be read either resources-relative or absolute.
    :param format: the format sequence to process the mat data. See top file description.
    :return: A list of mat contents in the specified format order
    """
    if not os.path.isfile(path):
        tpath = resources_path(path)
        if not os.path.isfile(path):
            raise IOError("Unable to find file.\nTried paths:\n%s\n%s" % (path, tpath))
        path = tpath

    data = scio.loadmat(path)
    res = []
    for (matkey, consumer) in format:
        res.append(consumer(data[matkey]))
    return res


def _read_frame_batch(frames: list, *, format):
    """
    Read a batch of frames according to the specified format.
    Use this for efficient allocation of memory.
    :param frames: the list of frames to be read and put into a single batch
    :param format: the format sequence to process the mat data. See top file description.
    :return: a list of mat contents in the specified format order, each organized in batches
    """
    frame_count = len(frames)
    assert frame_count > 0

    f1_idx = 0
    f1 = None
    while f1_idx < frame_count and f1 is None:
        try:
            f1 = _read_frame(frames[f1_idx], format=format)
        except SkipFrameException:
            f1 = None
            f1_idx += 1
    ret = [[] for _ in range(len(f1))]

    for idx in range(len(f1)):
        # ret[idx] = np.empty(shape=(frame_count,)+np.shape(f1[idx]), dtype=f1[idx].dtype)
        ret[idx].append(f1[idx])

    for frameidx in range(1+f1_idx, frame_count):
        try:
            fn = _read_frame(frames[frameidx], format=format)
        except SkipFrameException:
            continue
        for compidx in range(len(fn)):
            # ret[compidx][frameidx] = fn[compidx]
            ret[compidx].append(fn[compidx])

    # return ret
    return [np.array(comp, dtype=comp[0].dtype) for comp in ret]


def read_formatted_batch(frames: list, formatdict: dict):
    """
    Read a batch of frames and organize information into a dictionary of contents.

    This should not be used by training scripts directly,
    use a higher level DatasetManager instead.

    :param frames: the list of frames to be included into the desired batch
    :param formatdict: the format dictionary specifying a format sequence for each entry
                       of the resulting dictionary.
                       Sequences of data corresponding to one single key are concatenated
                       together along the last axis.

                       example:
                            framelist = ...
                            RGBDATA = ('img', lambda x:x)
                            DEPTHDATA = ('depth', lambda x:x)
                            HEATMAPDATA = ('heatm', lambda x:x)

                            // assume that .mat files have fields 'img', 'depth', 'heatm'
                            // and that they are channel-last format and do not need
                            // any preprocessing

                            fdict = {
                                'in': (RGBDATA, DEPTHDATA),
                                'out': (HEATMAPDATA,)
                                }
                            batchdict = read_formatted_batch(framelist, fdict)

                            batchdict now is:
                                {
                                    'in': <a batch of RGBD data len(framelist) long>
                                    'out': <a batch of heatmap data len(framelist) long>
                                }
    :return: a dictionary of formatted batches according to the formatdict specified.
    """
    mapping_dict = {}
    offset_counter = 0
    outformat = []
    for k in formatdict.keys():
        mapping_dict[k] = []
        for elem in formatdict[k]:
            mapping_dict[k].append(offset_counter)
            outformat.append(elem)
            offset_counter += 1
    data_elements = _read_frame_batch(frames, format=outformat)
    out = {}
    for k in mapping_dict.keys():
        elemlist = [data_elements[i] for i in mapping_dict[k]]
        out[k] = np.concatenate(elemlist, axis=-1)

    return out


if __name__ == '__main__':
    from matplotlib import pyplot as pplt
    from data.datasets.reading.formatting import *
    from library import *

    set_verbosity(DEBUG)
    frames = [crops_path("handsDondoo1_137.mat"), crops_path("handsBorgo2_1.mat")]
    dictformat = {
        'in': MIDFMT_CROP_RGBD,
        'out': MIDFMT_CROP_HEATMAP
    }

    data = read_formatted_batch(frames, formatdict=dictformat)
    pplt.imshow(data['in'][0, :, :, 0:3])
    pplt.show()
    pplt.imshow(data['out'][0, :, :, 0])
    pplt.show()