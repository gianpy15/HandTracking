import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..")))

from hand_data_management.utils import *

SERVER_SYMLINK = "/framedata"

if __name__ == '__main__':
    _, name = select_best_overall_frame()
    cache_frame(name)
    vidn = get_vidname(name)
    imgurl = os.path.join(SERVER_SYMLINK, vidn, "tmp", cached_frame_name(vidn, get_frameno(name)))
    print(imgurl)
    set_index_flag(get_index_from_frame(name),
                   flag=FLAG_PROCESSING,
                   idx=get_frameno(name))
