import numpy as np


def get_rotation_matrix(axis, angle):
    s = np.sin(angle)
    c = np.cos(angle)
    if axis == 0:
        # around x axis:
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])
    elif axis == 1:
        # around y axis:
        return np.array([[c, 0, -s],
                         [0, 1, 0],
                         [s, 0, c]])
    else:
        # around z axis:
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])
