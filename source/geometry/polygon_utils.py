import numpy as np
import matplotlib.pyplot as mplt
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
    npoints = 1000
    plist = np.random.uniform(low=low, high=high, size=(npoints, 2))
    poly = make_circle(resol=700,
                       center=[2, 0],
                       radius=2,
                       startdelta=0.3)

    def speedtest():
        for p in plist:
            is_inside(p, poly)

    print(timeit(speedtest, number=1))

    for p in plist:
        intest = is_inside(p, poly)
        col = "green" if intest else "red"
        mplt.plot(p[0], p[1], marker="o", color=col)

    mplt.fill(poly[:, 0], poly[:, 1], fill=False)
    mplt.show()