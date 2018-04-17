import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..")))

from data.datasets.framedata_management.naming import framebase
from data.datasets.framedata_management.utils import rewrite_labels
import re

CLEANFILE_RE = re.compile('^.*-[0-9]+\.txt$')
VIDEO_RE = re.compile('^[^.]+$')


def process_line(line, vidname):
    if line[0] == 'E':
        tokens = line.split(";")
        frameno = int(tokens[0][1:])
        labelinfo = tokens[1:]
        if len(labelinfo) != 63:
            print("WARNING: editing frame %d failed: incorrect number of tokens" % frameno)
            return False
        raw_labels = []
        for idx in range(21):
            raw_labels.append((float(labelinfo[3 * idx]),
                               float(labelinfo[3 * idx + 1]),
                               int(labelinfo[3 * idx + 2])))
        rewrite_labels(vidname, frameno, raw_labels)
    elif line[0] == 'D':
        frameno = int(line[1:])
        rewrite_labels(vidname, frameno)
    else:
        frameno = int(line)
        rewrite_labels(vidname, frameno)
    return True


if __name__ == '__main__':
    videoslist = [v for v in os.listdir(framebase) if re.match(VIDEO_RE, v)]
    freecleaners = [c for c in os.listdir(framebase) if re.match(CLEANFILE_RE, c)]

    for vidname in videoslist:
        videopath = os.path.join(framebase, vidname)
        cleanerlist = [os.path.join(framebase, vidname, f) for f in os.listdir(videopath) if re.match(CLEANFILE_RE, f)]
        for cleaner in cleanerlist:
            with open(cleaner, "r") as f:
                line = f.readline()
                while line != '':
                    process_line(line, vidname)
                    line = f.readline()
            os.remove(cleaner)

    for cleaner in freecleaners:
        vidname = cleaner.split("-")[0]
        cleaner = os.path.join(framebase, cleaner)
        with open(cleaner, "r") as f:
            line = f.readline()
            while line != '':
                process_line(line, vidname)
                line = f.readline()
        os.remove(cleaner)
