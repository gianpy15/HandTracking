import numpy as np
import os
import skvideo.io as skio
from image_loader.hand_io import *

BASECODE = 32
SEPARATOR = chr(126)
CONTRACTION = 4

FLAG_LABELED = '2'
FLAG_PROCESSING = '1'
FLAG_UNLABELED = '0'

framebase = pm.resources_path("framedata")
contributors = pm.resources_path("framedata/contributors.txt")


def encode_image(img):
    res = "%dx%dx%d%s" % (img.shape[0], img.shape[1], img.shape[2], SEPARATOR)
    for row in img:
        for pix in row:
            for ch in pix:
                res += chr(ch // CONTRACTION + BASECODE)
    return res


def decode_image(base):
    header, data = base.split(SEPARATOR)
    h, w, c = header.split('x')
    h = int(h)
    w = int(w)
    c = int(c)
    img = np.empty(shape=(h, w, c), dtype=np.uint8)
    idx = 0
    for row in range(h):
        for pix in range(w):
            for ch in range(c):
                img[row][pix][ch] = (ord(data[idx]) - BASECODE) * CONTRACTION
                idx += 1
    return img


def build_frame_root_from_vid(videopath):
    videopath = pm.resources_path(videopath)
    videoname = videopath.split('/')[-1].split('.')[0]
    framesdir = os.path.join(framebase, videoname)

    if os.path.exists(framesdir):
        print("%s ALREADY EXISTS. VIDEO WILL NOT BE EXPANDED." % framesdir)
        return

    videodata = skio.vread(pm.resources_path(videopath))
    os.makedirs(framesdir)
    for frameidx in range(len(videodata)):
        store(os.path.join(framesdir, frame_name(videoname, frameidx)), videodata[frameidx])
    build_empty_index_file(os.path.join(framesdir, index_name(videoname)), len(videodata))


def build_empty_index_file(complete_filename, index_len):
    f = open(complete_filename, "x")
    f.write(FLAG_UNLABELED * index_len)
    f.close()


def set_index_flag(complete_filename, flag, idx):
    f = open(complete_filename, "r+")
    f.seek(idx)
    f.write(flag)
    f.close()


def save_labels(labels, frame):
    vidn = get_vidname(frame)
    framedir = os.path.join(framebase, vidn)
    frame = os.path.join(framedir, frame)
    data = load(frame)
    store(frame, data=data, labels=labels)
    set_index_flag(os.path.join(framedir, index_name(vidn)),
                   flag=FLAG_LABELED,
                   idx=get_frameno(frame))


def select_best_frame(vidname):
    viddir = os.path.join(framebase, vidname)
    index = open(os.path.join(viddir, index_name(vidname)), "r")
    flagset = index.read()
    index.close()

    best_start = flagset.find(FLAG_UNLABELED)
    if best_start == -1:
        return -1, -1
    best_end = best_start
    current_start = best_start
    current_end = best_start + 1
    while current_end < len(flagset):
        if flagset[current_end] != FLAG_UNLABELED:
            if current_end - current_start - 1 > best_end - best_start:
                best_end = current_end - 1
                best_start = current_start
            while current_end < len(flagset) and flagset[current_end] != FLAG_UNLABELED:
                current_end += 1
            current_start = current_end
        current_end += 1
    if current_end - current_start - 1 > best_end - best_start:
        best_end = current_end - 1
        best_start = current_start
    return (best_end + best_start) // 2, best_end - best_start + 1


def select_best_overall_frame():
    vids = os.listdir(framebase)
    bestvid = ('', -1, -1)
    for vidname in vids:
        selected_frame, frames_interval = select_best_frame(vidname)
        if frames_interval > bestvid[2]:
            bestvid = (vidname, selected_frame, frames_interval)
    if bestvid[2] == -1:
        return None
    selected_vid_dir = os.path.join(framebase, bestvid[0])
    selected_frame_name = os.path.join(selected_vid_dir,
                                       frame_name(bestvid[0],
                                                  bestvid[1]))
    return load(selected_frame_name), frame_name(bestvid[0],
                                                 bestvid[1])


def add_contributor(nick):
    contribs = open(contributors, "r+")
    c = contribs.readline()
    pos = len(c)
    while c != '' and c.split(' ')[0] != nick:
        c = contribs.readline()
        pos += len(c)

    if c == '':
        contribs.write(nick + " 0001\n")
    else:
        current_amount = int(c.split(' ')[1])
        contribs.seek(pos-len(c))
        contribs.write(nick + " %04d" % (current_amount + 1,))


def frame_name(vidname, frameno):
    return "%s%04d.mat" % (vidname, frameno)


def index_name(vidname):
    return "%s-index.txt" % (vidname,)


def get_frameno(filename):
    return int(filename[-8:-4])


def get_vidname(filename):
    return filename.split("/")[-1][0:-8]


def get_index_from_frame(framename):
    vidname = get_vidname(framename)
    viddir = os.path.join(framebase, vidname)
    return os.path.join(viddir, index_name(vidname))


def get_complete_frame_path(framename):
    vidname = get_vidname(framename)
    return os.path.join(os.path.join(framebase, vidname), framename)
