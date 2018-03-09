import numpy as np


def blue_detect(pix):
    if pix[2] / pix[0] > 1.5 and pix[2] / pix[1] > 1.5:
        return True
    return False


def linesearch(frame, start, incr, value):
    curr = start
    while frame[curr[0], curr[1]] != value:
        curr = curr + incr
    return curr


def make_bluescreen_filter(rgb, color_detection=blue_detect):

    height, width = np.shape(rgb)[0:2]

    filter = np.empty(shape=np.shape(rgb)[0:2], dtype=np.bool)
    for row in range(len(rgb)):
        for pix in range(len(rgb[row])):
            filter[row, pix] = color_detection(rgb[row, pix])

    corners = []
    corners.append(linesearch(frame=filter,
                              start=np.array([0, 0]),
                              incr=np.array([1, 1]),
                              value=True))
    corners.append(linesearch(frame=filter,
                              start=np.array([0, width - 1]),
                              incr=np.array([1, -1]),
                              value=True))
    corners.append(linesearch(frame=filter,
                              start=np.array([height - 1, 0]),
                              incr=np.array([-1, 1]),
                              value=True))
    corners.append(linesearch(frame=filter,
                              start=np.array([height - 1, width - 1]),
                              incr=np.array([-1, -1]),
                              value=True))

    leftbound = max(corners[0][1], corners[2][1])
    rightbound = min(corners[1][1], corners[3][1])
    topbound = max(corners[0][0], corners[1][0])
    bottombound = min(corners[2][0], corners[3][0])

    for row in range(len(rgb)):
        for pix in range(len(rgb[row])):
            if pix > rightbound or pix < leftbound:
                filter[row, pix] = True
            elif row > bottombound or row < topbound:
                filter[row, pix] = True

    return filter
