from library.geometry.formatting import *
from library.geometry.transforms import *


def extract_prototype(imagepts, cal):
    if not all([pt.visible for pt in imagepts]):
        return None
    pts_cloud = np.array([p.to_camera_model(calibration=cal).as_row() for p in imagepts])
    pts_cloud = hand_format(pts_cloud - pts_cloud[0])

    rot1 = get_mapping_rot(pts_cloud[INDEX][3], [1, 0, 0])
    pts_cloud = hand_format([rot1 @ joint for joint in raw(pts_cloud)])
    angle = np.arctan(pts_cloud[BABY][3][2] / pts_cloud[BABY][3][1])
    rot2 = get_rotation_matrix([1, 0, 0], angle)
    pts_cloud = hand_format([rot2 @ joint for joint in raw(pts_cloud)])
    return pts_cloud


