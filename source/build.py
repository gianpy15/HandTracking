import sys
import os
import grp
sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..")))

import hand_data_management.utils as ut
from image_loader.hand_io import pm
import re


def is_vid_condition(vname):
    if os.path.isdir(vname):
        return False
    if os.path.splitext(vname)[1] != '.mp4':
        return False
    return True


def is_rawdir_condition(vname):
    if not os.path.isdir(vname):
        return False
    rgbnum = len([rgbfile for rgbfile in os.listdir(vname) if re.match('\d+\.rgb', rgbfile)])
    if rgbnum == 0:
        return False
    z16num = len([z16file for z16file in os.listdir(vname) if re.match('\d+\.z16', z16file)])
    if rgbnum != z16num:
        return False
    return True

vid_dir = pm.resources_path("vids")
VIDS = [os.path.join("vids", v) for v in os.listdir(vid_dir) if is_vid_condition(os.path.join(vid_dir, v))]

rawcam_dir = pm.resources_path("rawcam")
print(rawcam_dir)
print(os.listdir(rawcam_dir))
RAWS = [os.path.join("rawcam", v) for v in os.listdir(rawcam_dir) if is_rawdir_condition(os.path.join(rawcam_dir, v))]

wwwgrp = None

for g in grp.getgrall():
    if g[0] == 'www-data':
        wwwgrp = g[2]
        break


def edit_permissions(filename):
    os.chmod(filename, 0o775)
    os.chown(filename, wwwgrp, wwwgrp)


if __name__ == '__main__':
    if os.getuid() != 0:
        print("Please use sudo. I need to edit permissions.")
        exit(-1)

    for v in VIDS:
        vidname = os.path.split(v)[1]
        print("Processing video %s..." % vidname)
        if ut.build_frame_root_from_vid(v, post_process=edit_permissions):
            print("Successfully built %s video filesystem." % vidname)
        else:
            print("Video %s will not be expanded as it is already present." % vidname)

    for v in RAWS:
        vidname = os.path.split(v)[1]
        print("Processing rawcam video %s..." % vidname)
        if ut.build_frame_root_from_rawcam(v, post_process=edit_permissions):
            print("Successfully built %s video filesystem." % vidname)
        else:
            print("Video %s will not be expanded as it is already present." % vidname)

