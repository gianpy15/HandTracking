import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..")))

from data.datasets.framedata_management.index import *

if __name__ == '__main__':
    frame = sys.argv[1]
    vidname = get_vidname(frame)
    print(get_index_content(vidname))
