from hand_data_management.naming import *
from image_loader.hand_io import *
import os


def load_labeled_video(vidname):
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
    for frame in frames:
        fdata, ldata = load(frame)
        frame_data.append(fdata)
        label_data.append(ldata)
    if label_data[0] is None or label_data[-1] is None:
        return None
    linear_fill(label_data)
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
            print("%d %d" % (idxstart, idxend))
            for idxcur in range(idxstart + 1, idxend):
                startcoeff -= incr
                print(idxcur)
                label_data[idxcur] = label_data[idxstart] * startcoeff \
                                   + label_data[idxend] * (1 - startcoeff)
            idxstart = idxend
            idxend += 1
    return label_data

