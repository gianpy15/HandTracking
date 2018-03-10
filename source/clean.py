import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..")))

from hand_data_management.naming import framebase
from hand_data_management.utils import clear_labels
import re

CLEANFILE_RE = re.compile('^.*[0-9]\.txt$')
VIDEO_RE = re.compile('^[^.]+$')

if __name__ == '__main__':
    videoslist = [v for v in os.listdir(framebase) if re.match(VIDEO_RE, v)]
    for vidname in videoslist:
        videopath = os.path.join(framebase, vidname)
        cleanerlist = [os.path.join(framebase, vidname, f) for f in os.listdir(videopath) if re.match(CLEANFILE_RE, f)]
        for cleaner in cleanerlist:
            with open(cleaner, "r") as f:
                line = f.readline()
                while line != '':
                    frameno = int(line)
                    clear_labels(vidname, frameno)
                    line = f.readline()
            os.remove(cleaner)
