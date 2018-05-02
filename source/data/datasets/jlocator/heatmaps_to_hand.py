from library.geometry.formatting import *
import numpy as np


def heatmaps_to_hand(joints: np.ndarray, visibility: np.ndarray):
    coords = []
    heatshape = np.shape(joints)
    for idx in range(len(visibility)):
        peak = np.unravel_index(np.argmax(joints[:, :, idx]), heatshape[:2])
        coords.append([peak[0]/heatshape[0], peak[1]/heatshape[1], visibility[idx]])
    return hand_format(coords)

