import skvideo.io as skio
from os.path import *
import os
import re
import numpy as np
from image_loader.hand_io import *
import hand_data_management.grey_to_redblue_codec as gtrbc

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

    vid_data = gtrbc.codec(np.array(vid_data, dtype=np.long))

    skio.vwrite(join(VIDDIR, videoname), vid_data)

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
        read_depth_video("depthspeedtest_acc.mp4",
                     join("/home", "luca", "Scrivania", "rawcam", "out-1519566096"))
    print(timeit(action, number=1))
