import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..")))

from data.datasets.framedata_management.utils import *

if __name__ == '__main__':
    videos = os.listdir(framebase)
    videos = [nm for nm in videos if os.path.isdir(os.path.join(framebase, nm))]
    compls = {}
    for vid in videos:
        idxcontent = get_index_content(vid)
        count = 0
        for i in range(len(idxcontent)):
            if idxcontent[i] == FLAG_LABELED:
                count += 1
        completion = count / len(idxcontent)
        compls[vid] = completion

    compls_sorted = sorted(compls, key=compls.__getitem__, reverse=True)
    for vid in compls_sorted:
        print("Video completion for {}: {:.2f}%".format(vid, compls[vid]*100))
