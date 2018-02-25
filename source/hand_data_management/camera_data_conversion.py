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


def read_depth_video(videoname, framesdir, shape=(640, 480, 1), dtype=np.uint16):
    def z16_std_naming_frameno(filename):
        return int(splitext(split(filename)[1])[0])

    vid_data = read_frame_data(framesdir,
                               framenofunction=z16_std_naming_frameno,
                               shape=shape,
                               dtype=dtype,
                               framesregex="\d+\.z16")

    vid_data = grey_to_redblue_codec(vid_data, brange=2 ** 16 - 1)

    skio.vwrite(join(VIDDIR, videoname), vid_data)


def grey_to_redblue_codec(vid, brange):
    mult = np.pi / 2

    def to_redblue_codec_pix(pix):
        rate = pix[0] / brange
        if rate == 0:
            return [.0, .0, .0]
        return [255 * np.cos(rate * mult), .0, 255 * np.sin(rate * mult)]

    return np.apply_along_axis(to_redblue_codec_pix, axis=3, arr=vid)


if __name__ == '__main__':
    # produce a test video to check whether the greyscale to red-blue codec is fine
    vid_data = np.reshape(np.arange(start=0, stop=2 ** 16 -1, step=(2**16-1)/(50*150*150)), (50, 150, 150, 1))
    rb_vid_data = grey_to_redblue_codec(vid_data, 2 ** 16 -1)

    skio.vwrite(join(VIDDIR, "depthtest.mp4"), rb_vid_data)