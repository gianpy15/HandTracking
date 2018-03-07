import skvideo.io as skio
from os.path import *
import os
import re
import numpy as np
from image_loader.hand_io import *
# import hand_data_management.grey_to_redblue_codec as gtrbc

VIDDIR = pm.resources_path("vids")


def z16_std_naming_frameno(filename):
    return int(splitext(split(filename)[1])[0])


def rgb_std_naming_frameno(filename):
    return int(splitext(split(filename)[1])[0])


def read_frame_data(framesdir, framenofunction, shape, dtype=np.uint8, framesregex="."):
    if not isdir(framesdir):
        # print("Error: attempting to read frame data from non directory")
        # print("Provided pathname is %s" % framesdir)
        raise FileNotFoundError
    video_data = []
    for file in os.listdir(framesdir):
        if not re.match(framesregex, file) or isdir(file):
            continue
        file = join(framesdir, file)
        frame = np.reshape(np.fromfile(file, dtype=dtype), shape)
        video_data.append((frame, framenofunction(file)))
    if len(video_data) == 0:
        raise FileNotFoundError
    video_data.sort(key=lambda e: e[1])
    return np.array([frame for (frame, frameno) in video_data])


def default_read_rgb_args(framesdir, shape=(480, 640, 3), dtype=np.uint8):
    return {
        'framesdir': framesdir,
        'framenofunction': rgb_std_naming_frameno,
        'shape': shape,
        'dtype': dtype,
        'framesregex': "\d+\.rgb"
    }


def default_read_z16_args(framesdir, shape=(480, 640, 1), dtype=np.uint16):
    return {
        'framesdir': framesdir,
        'framenofunction': z16_std_naming_frameno,
        'shape': shape,
        'dtype': dtype,
        'framesregex': "\d+\.z16"
    }


def read_rgb_video(videoname, framesdir, shape=(480, 640, 3), dtype=np.uint8):
    vid_data = read_frame_data(**default_read_rgb_args(framesdir,
                                                       shape=shape,
                                                       dtype=dtype))
    skio.vwrite(join(VIDDIR, videoname), vid_data)


def read_depth_video(videoname, framesdir, shape=(480, 640, 1), dtype=np.uint16):
    vid_data = read_frame_data(**default_read_z16_args(framesdir,
                                                       shape=shape,
                                                       dtype=dtype))

    vid_data = grey_to_redblue_codec(np.array(vid_data, dtype=np.long))

    skio.vwrite(join(VIDDIR, videoname), vid_data)


def read_mesh_video(videoname, framesdir, shape=(480, 640), dtypergb=np.uint8, dtypedepth=np.uint16):
    rgb_data = read_frame_data(framesdir,
                               framenofunction=rgb_std_naming_frameno,
                               shape=shape + (3,),
                               dtype=dtypergb,
                               framesregex="\d+\.rgb") * 0.5
    depth_data = read_frame_data(framesdir,
                                 framenofunction=z16_std_naming_frameno,
                                 shape=shape + (1,),
                                 dtype=dtypedepth,
                                 framesregex="\d+\.z16") * 0.5

    depth_data = grey_to_redblue_codec(np.array(depth_data, dtype=np.long))

    vid_data = depth_data + rgb_data

    skio.vwrite(join(VIDDIR, videoname), vid_data)


def enhance_depth_vid(vid,  reduction=3, topval=255, flatten=False):
    # take video stats
    nonzero = np.count_nonzero(vid)
    total = vid.shape[0] * vid.shape[1] * vid.shape[2]
    avg = np.average(vid)
    var = np.var(vid)
    # correct video stats on non-zero values only
    corrected_avg = avg * total / nonzero
    corrected_var = (var + avg ** 2) * total / nonzero - corrected_avg ** 2
    # compute the bit range
    delta = reduction * np.sqrt(corrected_var)
    brange = (max(0, int(corrected_avg - delta)), int(corrected_avg + delta))
    reduced = vid - brange[0]
    if brange[0] > 0:
        reduced = reduced * (reduced < np.zeros(shape=(1,), dtype=vid.dtype) - brange[0])
    ret = reduced * (topval / brange[1])
    if flatten:
        ret = np.average(ret, axis=3)
    return ret


# Deprecated version of the codec. C++ compiled version is more than 100x faster.
def grey_to_redblue_codec(vid):
    mult = np.pi / 2
    nonzero = np.count_nonzero(vid)
    total = vid.shape[0] * vid.shape[1] * vid.shape[2]
    avg = np.average(vid)
    var = np.var(vid)
    corrected_avg = avg * total / nonzero
    corrected_var = (var + avg ** 2) * total / nonzero - corrected_avg ** 2
    brange = int(corrected_avg + 3 * np.sqrt(corrected_var))

    print("Building mapping buffer")
    mult /= brange
    pixmap = [[255 * np.cos(mult * i), .0, 255 * np.sin(mult * i)] for i in range(brange)]
    print("Mapping pixels...")

    def to_redblue_codec_pix(pix):
        if pix[0] == 0:
            return [.0, .0, .0]
        if pix[0] >= brange:
            p = brange - 1
        else:
            p = pix[0]
        return pixmap[p]

    video = []
    for idx in range(len(vid)):
        print("Processing frame %d/%d" % (idx, len(vid)))
        video.append(np.apply_along_axis(to_redblue_codec_pix, axis=2, arr=vid[idx]))
    return np.array(video, dtype=np.uint8)


from timeit import timeit

if __name__ == '__main__':
    def action():
        read_depth_video("hands_01depth.mp4",
                        join("/home", "luca", "Scrivania", "rawcam", "hands_01"))


    print(timeit(action, number=1))
