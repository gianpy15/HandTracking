from scipy.optimize import fsolve
from library.geometry.transforms import *
import numpy as np

# debug

# BASE PALM POINTS DIRECT INFERENCE


def line_projection_system(sols, params):
    return (sols[0] ** 2 - 2 * params["c01"] * sols[0] * sols[1] + sols[1] ** 2 - params["d01"] ** 2,
            sols[0] ** 2 - 2 * params["c02"] * sols[0] * sols[2] + sols[2] ** 2 - params["d02"] ** 2,
            sols[1] ** 2 - 2 * params["c12"] * sols[1] * sols[2] + sols[2] ** 2 - params["d12"] ** 2)


def line_projection_system_jac(sols, params):
    return [[2 * sols[0] - 2 * params["c01"] * sols[1],
            2 * sols[0] - 2 * params["c02"] * sols[2],
            0],
            [-2 * params["c01"] * sols[0] + 2 * sols[1],
             0,
             2 * sols[1] - 2 * params["c12"] * sols[2]],
            [0,
             -2 * params["c02"] * sols[0] + 2 * sols[2],
             -2 * params["c12"] * sols[1] + 2 * sols[2]]]


def get_points_projections_to_lines(basepts, lines, maxerr=1e-3, maxrestart=1000):
    params = {
        "c01": np.dot(lines[0], lines[1]),
        "c02": np.dot(lines[0], lines[2]),
        "c12": np.dot(lines[1], lines[2]),
        "d01": np.linalg.norm(basepts[0]-basepts[1]),
        "d02": np.linalg.norm(basepts[0]-basepts[2]),
        "d12": np.linalg.norm(basepts[1]-basepts[2])
    }

    dists = np.array([params["d01"], params["d02"], params["d12"]])
    cosines = np.array([params["c01"], params["c02"], params["c12"]])
    # avg = np.average(dists)
    # variance = np.var(dists)
    avg = dists[np.argmax(cosines)]/np.max(cosines)
    variance = np.var(dists / cosines)
    bestsol = None
    besterr = np.inf
    for i in range(maxrestart):
        start = np.random.normal(loc=avg, scale=variance, size=(3,))
        sol, info, stat, msg = fsolve(func=line_projection_system,
                                      x0=start,
                                      args=params,
                                      full_output=True)
        err = np.linalg.norm(info["fvec"])
        if err < maxerr:
            if sol[0] < 0:
                sol = -sol
            return np.array([lines[i] * sol[i] for i in range(3)])
        if err < besterr:
            if sol[0] < 0:
                bestsol = -sol
            else:
                bestsol = sol
    print("WARNING: Maxrestarts expired, the proposed solution has error %f" % besterr)
    return np.array([lines[i] * bestsol[i] for i in range(3)])


def get_points_projection_to_lines_pair(basepts, lines, maxerr=1e-3, maxrestart=10, maxtrials=10):
    pts = get_points_projections_to_lines(basepts=basepts, lines=lines)
    pts2 = get_points_projections_to_lines(basepts=basepts, lines=lines)

    count = 1
    while np.linalg.norm(pts - pts2) < 1e-5 and count < maxtrials:
        count += 1
        pts2 = get_points_projections_to_lines(basepts=basepts,
                                               lines=lines,
                                               maxerr=maxerr,
                                               maxrestart=maxrestart)
    return pts, pts2
