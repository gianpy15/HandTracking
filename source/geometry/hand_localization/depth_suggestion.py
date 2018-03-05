from geometry.formatting import *
from geometry.calibration import *
from geometry.transforms import normalize
from numpy.linalg import norm


def extract_model_info(image_joints, cal):
    """
    Separate depth information from line-wise information
    :param image_joints: the joints found in the image
    :param cal: the camera calibration to be used for mapping
    :return: (line, depth) standard dict-formatted hand info
            containing the model line versors and depth-complete points.
            If a point is not visible, its joint will be None.
    """
    if cal is None:
        cal = current_calibration
    vect_img_joints = raw(image_joints)
    depth_suggestion = hand_format([joint.to_camera_model(calibration=cal).as_row()
                                    if joint.visible and joint.depth > 0 else None
                                    for joint in vect_img_joints])
    line_suggestion = hand_format([normalize(joint.to_camera_model(calibration=cal).as_row())
                                   if joint.visible and joint.depth > 0 else
                                   joint.to_camera_model(calibration=cal).as_row()
                                   for joint in vect_img_joints])
    return line_suggestion, depth_suggestion


def depth_info_compare(measured, inferred, threshold, smooth=True):
    """
    Compare the depth-measured space point with the model-inferred one.
    We assume that the measure is less affected by noise, but may be completely wrong.
    The measure is accepted only if it agrees with the inferred value within a threshold.
    :param measured: the measured point in the point-cloud
    :param inferred: the model-inferred point to be compared with the measure
    :param threshold: the maximum allowed distance between the two to consider the measure reliable
    :param smooth: decide whether to use a smooth interpolation or a crisp choice
    :return: the chosen model point
    """
    # if no measurement is available...
    if measured is None:
        return inferred
    # else if the threshold has been passed:
    if smooth:
        rate = min(threshold, norm(inferred - measured)) / threshold
        return inferred * rate + measured * (1 - rate)
    if norm(inferred-measured) > threshold:
        return inferred
    return measured

