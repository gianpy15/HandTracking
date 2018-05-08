import numpy as np


def change_background(hand_img: np.ndarray, background_img: np.ndarray, filter):
    if hand_img.shape != background_img.shape:
        return
    for row in filter:
        for pixel in row:
            if pixel is True:
                pass


def mean_hsv(img: np.ndarray):
    pass
