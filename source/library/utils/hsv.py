from numba import jit, prange
import numpy as np


@jit(nopython=True, cache=True, nogil=True)
def rgb2hsv_pix(arr: np.ndarray):
    max = np.max(arr)
    min = np.min(arr)
    delta = max-min
    if delta == 0:
        h = 0
    elif max == arr[0]:
        h = (arr[1] - arr[2])/delta
    elif max == arr[1]:
        h = 2. + (arr[2] - arr[0])/delta
    else:
        h = 4. + (arr[0] - arr[1])/delta
    arr[0] = h / 6. % 1.

    if max == 0:
        arr[1] = 0
    else:
        arr[1] = delta / max
    arr[2] = max
    return arr


@jit(nopython=True, cache=True, nogil=True)
def hsv2rgb_pix(arr: np.ndarray):
    h6 = arr[0] * 6
    c = arr[1] * arr[2]
    x = c * (1 - np.abs(((h6 % 2) - 1)))
    m = arr[2] - c

    if h6 < 1:
        arr[0], arr[1], arr[2] = c + m, x + m, m
    elif h6 < 2:
        arr[0], arr[1], arr[2] = x + m, c + m, m
    elif h6 < 3:
        arr[0], arr[1], arr[2] = m, c + m, x + m
    elif h6 < 4:
        arr[0], arr[1], arr[2] = m, x + m, c + m
    elif h6 < 5:
        arr[0], arr[1], arr[2] = x + m, m, c + m
    else:
        arr[0], arr[1], arr[2] = c + m, m, x + m
    return arr


@jit(nopython=True, parallel=True, nogil=True)
def hsv2rgb(arr: np.ndarray):
    for i in prange(len(arr)):
        for j in prange(len(arr[i])):
            hsv2rgb_pix(arr[i, j])
    return arr


@jit(nopython=True, parallel=True, nogil=True)
def rgb2hsv(arr: np.ndarray):
    for i in prange(len(arr)):
        for j in prange(len(arr[i])):
            rgb2hsv_pix(arr[i, j])
    return arr
