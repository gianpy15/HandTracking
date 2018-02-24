import skvideo.io as skio
from os.path import *
import os
import re
import numpy as np
from image_loader.hand_io import *


VIDDIR = pm.resources_path("vids")


def read_frame_data(framesdir, framenofunction, shape, dtype=np.uint8, framesregex="."):
    if not isdir(framesdir):
        return None
    video_data = []
    for file in os.listdir(framesdir):
        if not re.match(framesregex, file) or isdir(file):
            continue
        frame = np.reshape(np.fromfile(file, dtype=dtype), shape)
        video_data.append((frame, framenofunction(file)))

    video_data.sort(key=lambda e: e[1])
    return np.array([frame for (frame, frameno) in video_data])


def read_rgb_video(videoname, framesdir, shape=(640, 480, 3), dtype=np.uint8):

    def rgb_std_naming_frameno(filename):
        return int(splitext(split(filename)[1])[0])

    vid_data = read_frame_data(framesdir,
                               framenofunction=rgb_std_naming_frameno,
                               shape=shape,
                               dtype=dtype,
                               framesregex="\d+\.rgb")
    skio.vwrite(join(VIDDIR, videoname), vid_data)


def read_depth_video(videoname, framesdir, shape=(640, 480, 2), dtype=np.uint8):

    def z16_std_naming_frameno(filename):
        return int(splitext(split(filename)[1])[0])

    vid_data = read_frame_data(framesdir,
                               framenofunction=z16_std_naming_frameno,
                               shape=shape,
                               dtype=dtype,
                               framesregex="\d+\.z16")

    vid_data = np.pad(vid_data, pad_width=((0, 0), (0, 0), (0, 0), (0, 1)), mode='constant')

    skio.vwrite(join(VIDDIR, videoname), vid_data)

