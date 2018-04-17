import numpy as np


# pythran export codec(int[][][][])
def codec(vid):
    mult = np.pi / 2
    nonzero = np.count_nonzero(vid)
    total = vid.shape[0] * vid.shape[1] * vid.shape[2]
    avg = np.average(vid)
    var = np.var(vid)
    corrected_avg = avg * total / nonzero
    corrected_var = (var + avg ** 2) * total / nonzero - corrected_avg ** 2
    brange = int(corrected_avg + 3 * np.sqrt(corrected_var))

    mult /= brange
    pixmap = np.array([[255 * np.cos(mult * i), .0, 255 * np.sin(mult * i)] for i in range(brange)], dtype=np.uint8)

    video = []
    for idx in range(len(vid)):
        print("Processing frame %d/%d" % (idx, len(vid)))
        frame = np.empty(shape=(vid.shape[1], vid.shape[2], 3), dtype=np.uint8)
        # print("Allocated frame with shape %f" % str(frame.shape))
        for rowidx in range(len(frame)):
            for pixidx in range(len(frame[rowidx])):
                tmp = vid[idx, rowidx, pixidx, 0]
                p = np.empty(shape=(3,), dtype=np.uint8)
                if tmp == 0:
                    p = np.array([0, 0, 0])
                elif tmp >= brange:
                    p = np.array(pixmap[-1])
                else:
                    p = np.array(pixmap[tmp])
                frame[rowidx, pixidx] = p[:]
        video.append(frame)
    return np.array(video, dtype=np.uint8)
