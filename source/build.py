import sys
import os
import grp
sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..")))

import hand_data_management.utils as ut
from image_loader.hand_io import pm


def is_vid_condition(vname):
    if os.path.isdir(vname):
        return False
    if os.path.splitext(vname)[1] != '.mp4':
        return False
    return True


vid_dir = pm.resources_path("vids")
VIDS = [os.path.join("vids", v) for v in os.listdir(vid_dir) if is_vid_condition(v)]

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



