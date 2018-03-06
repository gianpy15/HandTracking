import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..")))

from hand_data_management.index import *

if __name__ == '__main__':
    frame = sys.argv[1]
    vidname = get_vidname(frame)
    frameno = get_frameno(frame)
    current_status = get_index_content(vidname)[frameno]
    if current_status != FLAG_LABELED:
        set_index_flag(get_index_from_vidname(vidname),
                       flag=FLAG_UNLABELED,
                       idx=frameno)
    uncache_frame(frame)
