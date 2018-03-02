from geometry.hand_localization.fingers import *
from geometry.hand_localization.parameters import *
from geometry.calibration import *
from geometry.hand_localization.depth_suggestion import extract_model_info, depth_info_compare
from geometry.numerical.palm_threepts_inference import get_points_projection_to_lines_pair
from geometry.transforms import get_rotation_matrix, get_mapping_rot, normalize, get_rotation_angle_around_axis
from numpy.linalg import norm


def compute_hand_world_joints(base_hand_model, image_joints, side=RIGHT, cal=None, executor=None):
    """
    Compute the hand model in the space
    :param base_hand_model: the standard hand prototype to use for interpolation
    :param image_joints: the image points of the joints
    :param side: one of RIGHT or LEFT, the side of the hand
    :param cal: the calibration of the image joints to use
    :param executor: the optional executor pool to use to try to accelerate the process
    :return: The space model of the hand
    """

    # First thing is: compute the world model correspondences of the image points
    # This way we have defined the lines that we have to interpolate
    if cal is None:
        cal = current_calibration
    lines_info, depth_info = extract_model_info(image_joints, cal)

    # for now we assume that no point is visible, so they are all directional versors.

    # the hand model is normalized with respect to the wrist
    base_hand_model = hand_format([elem - base_hand_model[WRIST][0] for elem in raw(base_hand_model)])

    # based on the palm plane, extract the transformation that maps the model to the lines
    # conserving the distances between all points.
    # this transformation provides exact correspondence between prototype and image
    tr, rotmat = get_best_first_transform(base_hand_model,
                                          lines_info=lines_info,
                                          depth_sugg=depth_info,
                                          side=side)

    # apply this first transformation to the whole model
    first_hand_model = hand_format([first_transform_point(rotmat, tr, p)
                                    for p in raw(base_hand_model)])

    # find out plausible positions for MIDDLE and RING joints
    # interpolating the known INDEX and BABY
    t_index = first_hand_model[INDEX][0][0] / lines_info[INDEX][0][0]
    t_baby = first_hand_model[BABY][0][0] / lines_info[BABY][0][0]
    t_middle = t_index * 2 / 3 + t_baby / 3
    t_ring = t_index / 3 + t_baby * 2 / 3
    first_hand_model[MIDDLE][0] = t_middle * lines_info[MIDDLE][0]
    first_hand_model[RING][0] = t_ring * lines_info[RING][0]

    # now correct them with some depth measured information
    depth_tolerance = norm(first_hand_model[INDEX][0] - first_hand_model[BABY][0]) / 3
    first_hand_model[MIDDLE][0] = depth_info_compare(inferred=first_hand_model[MIDDLE][0],
                                                     measured=depth_info[MIDDLE][0],
                                                     threshold=depth_tolerance)
    first_hand_model[RING][0] = depth_info_compare(inferred=first_hand_model[RING][0],
                                                   measured=depth_info[RING][0],
                                                   threshold=depth_tolerance)

    # now we have to face the fingers: start getting the palm direction
    palm_base_axis = normalize(np.cross(first_hand_model[INDEX][0] - first_hand_model[WRIST][0],
                                        first_hand_model[BABY][0] - first_hand_model[WRIST][0]))

    end_model = {WRIST: first_hand_model[WRIST]}

    if executor is None:
        # if no executor pool has been provided, take care of all the fingers yourself
        for finger in (INDEX, MIDDLE, RING, BABY):
            finger_position = compute_generic_finger(first_hand_model=first_hand_model,
                                                     palm_base_axis=palm_base_axis,
                                                     lines_info=lines_info,
                                                     finger=finger,
                                                     depth_sugg=depth_info)
            end_model[finger] = [first_hand_model[finger][0], 0, 0, 0]
            for idx in range(1, 4):
                end_model[finger][idx] = finger_position[idx - 1]

        # the thumb has special rotations that need peculiar attention:
        end_model[THUMB] = compute_thumb(first_hand_model=first_hand_model,
                                         palm_base_axis=palm_base_axis,
                                         lines_info=lines_info,
                                         depth_sugg=depth_info)
    else:
        # if some angel provided any executor, schedule the four fingers
        futures = {}
        task = compute_generic_finger_wrap(first_hand_model=first_hand_model,
                                           palm_base_axis=palm_base_axis,
                                           lines_info=lines_info,
                                           depth_sugg=depth_info)
        for finger in (INDEX, MIDDLE, RING, BABY):
            futures[finger] = executor.submit(task, finger)

        # and then solve the thumb problem
        end_model[THUMB] = compute_thumb(first_hand_model=first_hand_model,
                                         palm_base_axis=palm_base_axis,
                                         lines_info=lines_info,
                                         depth_sugg=depth_info)

        # finally harvest the results
        for finger in (INDEX, MIDDLE, RING, BABY):
            end_model[finger] = [first_hand_model[finger][0], 0, 0, 0]
            for idx in range(1, 4):
                end_model[finger][idx] = futures[finger].result()[idx - 1]

    # and finally we have it
    # end_check(base_hand_model, end_model)
    return end_model


def first_transform_point(rotmat, translation, point):
    """
    The first transformation done in the process of hand building.
    This is an affine transformation of the whole model
    """
    return (rotmat @ point) + translation


# TODO find some reasonable reaction to the depth suggestions
def get_best_first_transform(base_hand_model, lines_info, depth_sugg, side):
    """
    Extract the geometrical transformation that interpolates WRIST, INDEX base and BABY base
    with the given world model. Based on the suggested side, a RIGHT or LEFT hand is returned
    :param base_hand_model: the prototype of the hand model.
    :param lines_info: the model lines stating where all joints should lie
    :param depth_sugg: the set of the depth-wise measures to take into account
    :param side: RIGHT or LEFT, depending on the seen hand
    :return: a tuple of the form (tr, rotmat) containing the selected translation and rotation to
            be applied to the base hand model.
    """
    base_triangle_pts = [base_hand_model[WRIST][0], base_hand_model[INDEX][0], base_hand_model[BABY][0]]
    base_triangle_lines = [lines_info[WRIST][0], lines_info[INDEX][0], lines_info[BABY][0]]
    model1, model2 = get_points_projection_to_lines_pair(base_triangle_pts, base_triangle_lines)

    def get_transformation_from_model(model):
        # translation default to wrist
        tr = model[0]
        # rotate the index to align it to the model
        rotmat = get_mapping_rot(base_triangle_pts[1], model[1] - model[0])
        # find out the transformation to align the baby as well
        axis = normalize(model[1] - model[0])
        angle = get_rotation_angle_around_axis(axis,
                                               rotmat @ base_triangle_pts[2],
                                               model[2] - model[0])
        # correct the rotation to align all the three
        rotmat = np.matmul(get_rotation_matrix(axis, angle), rotmat)
        return tr, rotmat

    tr1, rotmat1 = get_transformation_from_model(model1)

    model_axis_unnorm = np.cross(model1[1] - model1[0], model1[2] - model1[0]) * side
    avg_fingers = np.array([0., 0., 0.])
    for finger in DIRECTION_REVEALING_FINGERS:
        base = first_transform_point(rotmat1, tr1, base_hand_model[finger][0])
        joints, _ = build_finger_fast(basejoint=base,
                                      lengths=[norm(base_hand_model[finger][idx + 1] - base_hand_model[finger][idx])
                                               for idx in range(SIDE_N_ESTIM)],
                                      jointversors=[lines_info[finger][idx + 1] for idx in range(SIDE_N_ESTIM)],
                                      depthsugg=depth_sugg[finger])
        avg_fingers = avg_fingers + np.average(joints, axis=0)
    avg_fingers = avg_fingers / len(FINGERS)

    if np.dot(avg_fingers - model1[0], model_axis_unnorm) > 0:
        return tr1, rotmat1
    return get_transformation_from_model(model2)
