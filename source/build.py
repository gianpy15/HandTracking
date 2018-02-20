import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..")))

import hand_data_management.utils as ut

VIDS = ["vids/snap.mp4"]

if __name__ == '__main__':
    for v in VIDS:
        ut.build_frame_root_from_vid(v)
