import numpy as np
import matplotlib.pyplot as mplt
from numba import jit, prange
from timeit import timeit


def is_inside(point, polygon):
    angle = 0.
    for idx in range(len(polygon)-1):
        angle += compute_angle(polygon[idx], point, polygon[idx+1])
    angle += compute_angle(polygon[len(polygon)-1], point, polygon[0])
    return angle > np.pi or angle < -np.pi


def compute_angle(p1, p2, p3):
    p1n = p1-p2
    p3n = p3-p2
    anglesin = (p1n[0]*p3n[1]-p1n[1]*p3n[0])/(np.linalg.norm(p1n)*np.linalg.norm(p3n))
    return np.arcsin(anglesin)


# ##################### ARCSIN TABLE #######################


@jit
def make_arcsin_table(resolution):
    x = np.arange(-1.0, 1.0, 2 / resolution)
    y = np.arcsin(x)
    return np.array(y)


@jit(nopython=False, cache=True)
def table_asin(value, table):
    return table[int((value + 1.) * (len(table)//2))]

# #################### FASTER VERSION ####################


@jit(nopython=False, cache=True)
def fast_is_inside(point, polygon):
    return contains_origin(polygon - point)


@jit(nopython=False, cache=True)
def contains_origin(polygon):
    # normalizing once for all times
    norm_poly = np.empty(shape=(len(polygon), 2), dtype=np.float32)
    for idx in prange(len(polygon)):
        norm = np.linalg.norm(polygon[idx])
        if norm == 0:
            return True
        norm_poly[idx] = polygon[idx] / np.linalg.norm(polygon[idx])

    # computing angles WRT origin
    angle = 0.
    for idx in prange(len(norm_poly)-1):
        angle += fast_compute_angle(norm_poly[idx], norm_poly[idx+1])
    angle += fast_compute_angle(norm_poly[len(norm_poly)-1], norm_poly[0])
    return angle > np.pi or angle < -np.pi


# asintable = make_arcsin_table(5000)

@jit
def fast_compute_angle(p1, p2):
    return np.arcsin(p1[0]*p2[1] - p1[1]*p2[0])

# #################### TEST UTILS ##########################


def make_circle(resol, center, radius, startdelta=0.):
    delta = np.pi * 2 / resol
    angle = startdelta
    ret = np.empty(shape=(resol, 2))
    for idx in range(resol):
        ret[idx] = rotate([radius, 0], angle) + center
        angle += delta
    return ret


def rotate(point, angle):
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.array([cos*point[0] - sin*point[1], sin*point[0] + cos*point[1]])


if __name__ == '__main__':
    low = -5
    high = 5
    npoints = 200
    plist = np.random.uniform(low=low, high=high, size=(npoints, 2))
    poly = make_circle(resol=200,
                       center=[2, 0],
                       radius=2,
                       startdelta=0.3)
    if True:
        def speedtest():
            for idx in range(len(plist)):
                fast_is_inside(plist[idx], poly)

        print(timeit(speedtest, number=1))
        for p in plist:
            intest = fast_is_inside(p, poly)
            col = "green" if intest else "red"
            mplt.plot(p[0], p[1], marker="o", color=col)

        mplt.fill(poly[:, 0], poly[:, 1], fill=False)
        mplt.show()
    else:
        resol = 5000
        table = make_arcsin_table(resol)
        x = np.arange(-1, 1, 0.000001)

        def asin_speed_test():
            for s in x:
                np.arcsin(s)

        def table_speed_test():
            for s in x:
                table_asin(s, table)

        print(timeit(asin_speed_test, number=1))
        print(timeit(table_speed_test, number=1))
        mplt.plot(x, np.arcsin(x), color='green')
        mplt.plot(x, [table_asin(s, table) for s in x], color='blue')
        mplt.show()
