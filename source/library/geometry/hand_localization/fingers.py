from geometry.hand_localization.parameters import *
from geometry.numerical.finger_cone_sectors import *
from geometry.transforms import get_rotation_matrix, get_mapping_rot, get_rotation_angle_around_axis
from geometry.hand_localization.depth_suggestion import depth_info_compare


# ################################## LOW LEVEL UTILS ############################


def cone_project(axis, vec, coneangle):
    """
    Compute the projection of vector vec with respect to the cone having axis as base axis
    and with aperture coneangle
    """
    rotangle = max(0, np.arccos(np.dot(axis, vec) / norm(vec)) - coneangle)
    if rotangle == 0:
        return vec
    rotax = normalize(np.cross(vec, axis))
    return get_rotation_matrix(rotax, rotangle) @ vec


def sphere_line_intersection(center, radius, line):
    # <tv-c, tv-c> = r2
    # t2 -2<v, c>t + <c, c> - r2 = 0
    bh = - np.dot(line, center)
    delta = bh ** 2 - np.dot(center, center) + radius ** 2
    if delta < 0:
        sol = - bh * line
        return sol, sol, norm(np.cross(sol, line)) * 1e+20, True
    return (np.sqrt(delta) - bh) * line, (-bh - np.sqrt(delta)) * line, 0, False


# ################################### LOW LEVEL FINGER COMPUTATION ################################


def build_finger_num(basejoint, lengths, jointversors, depthsugg, config=None):
    if len(lengths) == 0:
        return []

    def compute_config(proposed_sol):
        if config is None:
            return None
        rotmat = get_mapping_rot(config[START_DIR], proposed_sol - basejoint)
        newconf = config.copy()
        newconf[START_DIR] = rotmat @ config[START_DIR]
        if config[THUMBAXIS] is not None:
            axisrot = get_rotation_angle_around_axis(axis=config[THUMBAXIS],
                                                     p1=config[START_DIR],
                                                     p2=proposed_sol - basejoint)
            rotmat = np.matmul(get_rotation_matrix(axis=newconf[START_DIR], angle=axisrot), rotmat)
        newconf[AROUND_DIR] = rotmat @ config[AROUND_DIR]
        newconf[NORM_DIR] = rotmat @ config[NORM_DIR]
        newconf[MAXANGLE] = config[MAXANGLE][1:]
        newconf[MAXWIDEANGLE] = config[MAXWIDEANGLE][1:]
        newconf[THUMBAXIS] = None
        return newconf

    numeric_suggest_list = build_finger_fast(basejoint=basejoint,
                                             lengths=[lengths[0]],
                                             jointversors=[jointversors[0]],
                                             depthsugg=[depthsugg[0]])[0]

    numeric_suggest_list.append(cone_project(axis=config[AROUND_DIR],
                                             vec=numeric_suggest_list[0] - basejoint,
                                             coneangle=config[MAXANGLE][0]) + basejoint)

    if depthsugg[0] is not None:
        numeric_suggest_list.append(depthsugg[0])
    joint = find_best_point_in_cone(center=basejoint,
                                    radius=lengths[0],
                                    objline=jointversors[0],
                                    norm_vers=config[AROUND_DIR],
                                    tang_vers=config[NORM_DIR],
                                    normcos=np.cos(config[MAXANGLE][0]),
                                    planecos=np.cos(config[MAXWIDEANGLE][0]),
                                    suggestion=numeric_suggest_list)

    joint = depth_info_compare(inferred=joint,
                               measured=depthsugg[0],
                               threshold=norm(joint - basejoint))

    return [joint] + build_finger_num(basejoint=joint,
                                      lengths=lengths[1:],
                                      jointversors=jointversors[1:],
                                      depthsugg=depthsugg[1:],
                                      config=compute_config(joint))


def build_finger_fast(basejoint, lengths, jointversors, depthsugg):
    if len(lengths) == 0:
        return [], 0

    p1, p2, loss, degen = sphere_line_intersection(center=basejoint,
                                                   radius=lengths[0],
                                                   line=jointversors[0])
    p1 = depth_info_compare(inferred=p1,
                            measured=depthsugg[0],
                            threshold=norm(p1 - basejoint))
    if degen:
        finger, moreloss = build_finger_fast(p1,
                                             lengths[1:],
                                             jointversors[1:],
                                             depthsugg[1:])
        return [p1] + finger, loss + moreloss
    finger1, moreloss1 = build_finger_fast(p1,
                                           lengths[1:],
                                           jointversors[1:],
                                           depthsugg[1:])
    if moreloss1 == 0:
        return [p1] + finger1, 0
    p2 = depth_info_compare(inferred=p2,
                            measured=depthsugg[0],
                            threshold=norm(p2 - basejoint))
    finger2, moreloss2 = build_finger_fast(p2,
                                           lengths[1:],
                                           jointversors[1:],
                                           depthsugg[1:])
    if moreloss2 < moreloss1:
        return [p2] + finger2, moreloss2
    return [p1] + finger1, moreloss1


# ####################################### HIGH LEVEL FINGER COMPUTATION ##################################

def compute_generic_finger_wrap(first_hand_model, palm_base_axis, lines_info, depth_sugg):
    return lambda f: compute_generic_finger(first_hand_model, palm_base_axis, lines_info, depth_sugg, finger=f)


def compute_generic_finger(first_hand_model, palm_base_axis, lines_info, depth_sugg, finger):
    # the build finger utility needs complex configuration, here we build it
    conf = {NORM_DIR: palm_base_axis,
            START_DIR: normalize(first_hand_model[finger][1] - first_hand_model[finger][0])
            }
    lengths = []
    conf[AROUND_DIR] = normalize(conf[START_DIR] * normdirrates[finger][START_DIR]
                                 + conf[NORM_DIR] * normdirrates[finger][NORM_DIR])
    conf[MAXANGLE] = maxangle[finger]
    conf[MAXWIDEANGLE] = maxwideangle[finger]
    conf[THUMBAXIS] = None
    for idx in range(3):
        lengths.append(norm(first_hand_model[finger][idx + 1] - first_hand_model[finger][idx]))
    finger_position = build_finger_num(basejoint=first_hand_model[finger][0],
                                       lengths=lengths,
                                       jointversors=lines_info[finger][1:],
                                       config=conf,
                                       depthsugg=depth_sugg[finger][1:])
    return finger_position


def compute_thumb(first_hand_model, palm_base_axis, lines_info, depth_sugg):
    conf = {NORM_DIR: palm_base_axis,
            START_DIR: normalize(first_hand_model[THUMB][1] - first_hand_model[WRIST][0])}
    end_model = [0, 0, 0, 0]
    thumb_rotation = get_rotation_matrix(-conf[START_DIR], angle=np.pi / 4)
    conf[NORM_DIR] = thumb_rotation @ palm_base_axis
    conf[AROUND_DIR] = normalize(conf[START_DIR] * normdirrates[THUMB][START_DIR]
                                 + conf[NORM_DIR] * normdirrates[THUMB][NORM_DIR])
    conf[MAXANGLE] = maxangle[THUMB]
    conf[MAXWIDEANGLE] = maxwideangle[THUMB]
    conf[THUMBAXIS] = normalize(first_hand_model[INDEX][0] - first_hand_model[WRIST][0])
    lines = [lines_info[THUMB][1]]
    lengths = [norm(first_hand_model[WRIST][0] - first_hand_model[THUMB][1])]
    for idx in range(1, 3):
        lengths.append(norm(first_hand_model[THUMB][idx + 1] - first_hand_model[THUMB][idx]))
        lines.append(lines_info[THUMB][idx + 1])
    thmbend = build_finger_num(basejoint=first_hand_model[WRIST][0],
                               lengths=lengths,
                               jointversors=lines,
                               config=conf,
                               depthsugg=depth_sugg[THUMB][1:])
    for idx in range(1, 4):
        end_model[idx] = thmbend[idx - 1]

        # the point zero of the thumb is not a real articulation,
    # for this reason it is artificially aligned to the rest
    align_mat = get_mapping_rot(-first_hand_model[WRIST][0] + first_hand_model[THUMB][1],
                                -first_hand_model[WRIST][0] + thmbend[0])
    thumb_zero = (align_mat @ (first_hand_model[THUMB][0] - first_hand_model[WRIST][0])) + first_hand_model[WRIST][0]
    end_model[0] = thumb_zero

    return end_model
