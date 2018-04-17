from data.datasets.framedata_management.naming import *
from data.datasets.io.hand_io import *
import numpy as np
import os

# define symbols for the gapflags
LABELED = 1
UNLABELED = 0


def load_labeled_video(vidname, getdepth=False, fillgaps=True, gapflags=False):
    """
    Load all frames of a video in a 4-dimentional numpy array, with all available labels.
    Missing labels are written linearly interpolating the available data.
    Usage example: frames, labels = load_labeled_video("snap")
    :param vidname: a string with the name of the video
    :return: a tuple with all frames and labels in the format (frames, labels)
    """
    dirpath = os.path.join(framebase, vidname)
    frames = [os.path.join(dirpath, fname)
              for fname in os.listdir(dirpath)
              if fname.split(".")[-1] == 'mat']
    frames.sort(key=get_frameno)
    frame_data = []
    label_data = []
    gap_list = []
    for frame in frames:
        if getdepth:
            fdata, ldata = load(frame, format=(DEPTH_DATA, LABEL_DATA))
        else:
            fdata, ldata = load(frame)
        frame_data.append(fdata)
        label_data.append(ldata)
        gap_list.append(UNLABELED if ldata is None else LABELED)
    frame_data = np.array(frame_data)
    label_data = np.array(label_data)
    gap_list = np.array(gap_list)
    if fillgaps:
        if label_data[0] is None or label_data[-1] is None:
            # unable to fill gaps, None labels are returned as a precaution
            if gapflags:
                # expecting three-tuple output
                return frame_data, None, gap_list
            return frame_data, None
        linear_fill(label_data)
    if gapflags:
        return frame_data, label_data, gap_list
    return frame_data, label_data


def linear_fill(label_data):
    idxstart = 0
    idxend = 1

    while idxend < len(label_data):
        while idxend < len(label_data) and label_data[idxend] is None:
            idxend += 1
        if idxend < len(label_data):
            rangelen = idxend - idxstart
            startcoeff = 1.0
            incr = 1./rangelen
            for idxcur in range(idxstart + 1, idxend):
                startcoeff -= incr
                label_data[idxcur] = label_data[idxstart] * startcoeff \
                                   + label_data[idxend] * (1 - startcoeff)
            idxstart = idxend
            idxend += 1
    return label_data

