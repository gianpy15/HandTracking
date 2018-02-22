import numpy as np
from scipy.optimize import fsolve
import timeit


def get_rotation_matrix(axis, angle):
    if isinstance(axis, (list, tuple, np.ndarray)):
        return get_rotation_matrix_from_quat(*get_quat_from_axis_angle(axis, angle))
    else:
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


def get_cross_matrix(v):
    return np.array([[0., -v[2], v[1]],
                     [v[2], 0., -v[0]],
                     [-v[1], v[0], 0.]])


def cross_product(v1, v2):
    mat = get_cross_matrix(v1)
    return np.reshape(np.matmul(mat, np.expand_dims(v2, 1)), (3,))


def get_rotation_matrix_from_quat(a, b, c, d):
    a2 = a * a
    b2 = b * b
    c2 = c * c
    d2 = d * d
    bc = 2 * b * c
    ad = 2 * a * d
    bd = 2 * b * d
    ac = 2 * a * c
    ab = 2 * a * b
    cd = 2 * c * d
    return np.array([[a2 + b2 - c2 - d2, bc - ad, bd + ac],
                     [bc + ad, a2 + c2 - b2 - d2, cd - ab],
                     [bd - ac, cd + ab, a2 + d2 - b2 - c2]])


def get_quat_from_axis_angle(axis, angle):
    semi = angle / 2
    normax = np.sin(semi) * axis / np.linalg.norm(axis)
    return np.cos(semi), normax[0], normax[1], normax[2]


def get_mapping_rot(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    axis = cross_product(v1, v2)
    return get_rotation_matrix(axis, angle=np.arccos(np.dot(v1, v2) / n1 / n2))


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
        sol, info, stat, msg = fsolve(line_projection_system, start, params, fprime=line_projection_system_jac, full_output=True)
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
