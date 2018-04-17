from geometry.calibration import ImagePoint
import numpy as np


def extract_imgpoints(labels, depthimg):
    out = []
    resx = np.shape(depthimg)[1]
    resy = np.shape(depthimg)[0]
    for (x, y, occluded) in labels:
        coords = x * resx, y * resy
        if not occluded:
            out.append(ImagePoint(coords=coords,
                                  depth=depthimg[int(coords[1]), int(coords[0]), 0]))
        else:
            out.append(ImagePoint(coords=coords))
    return np.array(out)
