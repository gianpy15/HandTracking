import numpy as np


def column_matmul(m, v):
    return np.array([dot(m[0], v), dot(m[1], v), dot(m[2], v)])


#pythran export dot(float[], float[])
def dot(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

#pythran export get_rotation_matrix(float[], float)
def get_rotation_matrix(axis, angle):
    args = get_quat_from_axis_angle(axis, angle)
    return get_rotation_matrix_from_quat(args[0], args[1], args[2], args[3])

#pythran export get_cross_matrix(float[])
def get_cross_matrix(v):
    return np.array([[0., -v[2], v[1]],
                     [v[2], 0., -v[0]],
                     [-v[1], v[0], 0.]])


#pythran export normalize(float[])
def normalize(vec):
    return vec / np.linalg.norm(vec)


#pythran export get_rotation_matrix_from_quat(float, float, float, float)
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


#pythran export get_quat_from_axis_angle(float[], float)
def get_quat_from_axis_angle(axis, angle):
    semi = angle / 2
    normax = np.sin(semi) * normalize(axis)
    return np.cos(semi), normax[0], normax[1], normax[2]


#pythran export get_mapping_rot(float[], float[])
def get_mapping_rot(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    axis = column_matmul(get_cross_matrix(v1), v2)
    if np.linalg.norm(axis) <= n1 * n2 * 1e-10:
        if dot(v1, v2) > 0:
            return np.eye(3)
        if v1[2] != 0:
            axis = normalize(column_matmul(get_cross_matrix(v1), np.array([1, 0, 0])))
            return get_rotation_matrix_from_quat(0, axis[0], axis[1], axis[2])
        else:
            axis = normalize(column_matmul(get_cross_matrix(v1), np.array([0, 0, 1])))
            return get_rotation_matrix_from_quat(0, axis[0], axis[1], axis[2])
    return get_rotation_matrix(axis, angle=np.arccos(dot(v1, v2) / n1 / n2))
