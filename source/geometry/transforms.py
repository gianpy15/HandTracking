import numpy as np


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


def normalize(vec):
    return vec / np.linalg.norm(vec)


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
    normax = np.sin(semi) * normalize(axis)
    return np.cos(semi), normax[0], normax[1], normax[2]


def get_mapping_rot(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    if np.linalg.norm(axis) <= n1 * n2 * 1e-10:
        if np.dot(v1, v2) > 0:
            return np.eye(3)
        if v1[2] != 0:
            axis = normalize(np.cross(v1, [1, 0, 0]))
            return get_rotation_matrix_from_quat(0, axis[0], axis[1], axis[2])
        else:
            axis = normalize(np.cross(v1, [0, 0, 1]))
            return get_rotation_matrix_from_quat(0, axis[0], axis[1], axis[2])
    return get_rotation_matrix(axis, angle=np.arccos(np.dot(v1, v2) / n1 / n2))


def get_rotation_angle_around_axis(axis, p1, p2):
    v1 = normalize(p1 - axis * np.dot(p1, axis))
    v2 = normalize(p2 - axis * np.dot(p2, axis))
    cross = np.cross(v1, v2)
    if np.dot(axis, cross) > 0:
        return np.arccos(np.dot(v1, v2))
    return -np.arccos(np.dot(v1, v2))

