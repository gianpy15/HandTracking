import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..")))

from php_called_scripts.utils import *

if __name__ == '__main__':
    labels = sys.argv[1]
    frame = sys.argv[2]
    tokens = labels.split(',')
    if len(tokens) != 63:
        exit(-1)
    raw_labels = []
    for idx in range(21):
        raw_labels.append((float(tokens[3*idx]),
                           float(tokens[3*idx+1]),
                           1 if tokens[3*idx+2] in ('true', 'True', 'TRUE') else 0))
    save_labels(labels=raw_labels, frame=frame)
    if len(sys.argv) > 2:
        add_contributor(sys.argv[3].replace(" ", ""))
    else:
        add_contributor("Anonymous")

