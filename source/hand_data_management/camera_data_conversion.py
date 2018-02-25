import skvideo.io as skio
from os.path import *
import os
import re
import numpy as np
from image_loader.hand_io import *

VIDDIR = pm.resources_path("vids")


def read_frame_data(framesdir, framenofunction, shape, dtype=np.uint8, framesregex="."):
    if not isdir(framesdir):
        print("Warning: attempting to read frame data from non directory")
        print("Provided pathname is %s" % framesdir)
        return None
    video_data = []
    for file in os.listdir(framesdir):
        if not re.match(framesregex, file) or isdir(file):
            continue
        file = join(framesdir, file)
        frame = np.reshape(np.fromfile(file, dtype=dtype), shape)
        video_data.append((frame, framenofunction(file)))

    video_data.sort(key=lambda e: e[1])
    return np.array([frame for (frame, frameno) in video_data])


def read_rgb_video(videoname, framesdir, shape=(480, 640, 3), dtype=np.uint8):
    def rgb_std_naming_frameno(filename):
        return int(splitext(split(filename)[1])[0])

    vid_data = read_frame_data(framesdir,
                               framenofunction=rgb_std_naming_frameno,
                               shape=shape,
                               dtype=dtype,
                               framesregex="\d+\.rgb")
    skio.vwrite(join(VIDDIR, videoname), vid_data)


def read_depth_video(videoname, framesdir, shape=(480, 640, 1), dtype=np.uint16):
    def z16_std_naming_frameno(filename):
        return int(splitext(split(filename)[1])[0])

    vid_data = read_frame_data(framesdir,
                               framenofunction=z16_std_naming_frameno,
                               shape=shape,
                               dtype=dtype,
                               framesregex="\d+\.z16")

    vid_data = grey_to_redblue_codec(vid_data)

    skio.vwrite(join(VIDDIR, videoname), vid_data)


def grey_to_redblue_codec(vid):
    mult = np.pi / 2
    nonzero = np.count_nonzero(vid)
    total = vid.shape[0]*vid.shape[1]*vid.shape[2]
    avg = np.average(vid)
    var = np.var(vid)
    corrected_avg = avg * total / nonzero
    corrected_var = (var + avg ** 2) * total / nonzero - corrected_avg ** 2
    brange = corrected_avg + 3 * np.sqrt(corrected_var)

    def to_redblue_codec_pix(pix):
        rate = pix[0] / brange
        if rate > 1.0:
            rate = 1.0
        if rate == 0:
            return [.0, .0, .0]
        return [255 * np.cos(rate * mult), .0, 255 * np.sin(rate * mult)]

    return np.array(np.apply_along_axis(to_redblue_codec_pix, axis=3, arr=vid), dtype=np.uint8)


if __name__ == '__main__':
    read_depth_video("depthtest.mp4",
                     join("/home", "luca", "Scrivania", "rawcam", "out-1519561220"))