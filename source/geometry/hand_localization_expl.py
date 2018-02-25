from geometry.calibration import *
from geometry.numerical import *
from geometry.formatting import *
from geometry.transforms import *
from geometry.default_model_loading import build_default_hand_model
from numpy.linalg import norm
import timeit
import concurrent.futures as fut

# Maximum rotation angles of any joints around the median rotation axis
maxangle = {
    THUMB: np.pi / 180 * 35,
    INDEX: np.pi / 180 * 55,
    MIDDLE: np.pi / 180 * 50,
    RING: np.pi / 180 * 46,
    BABY: np.pi / 180 * 47
}

# The angles defining where is the median rotation axis of the finger.
# Median rotation is the one pointing at this angle in between the
# palm and the finger direction
normdirangles = {
    THUMB: np.pi / 180 * 45,
    INDEX: np.pi / 180 * 45,
    MIDDLE: np.pi / 180 * 45,
    RING: np.pi / 180 * 45,
    BABY: np.pi / 180 * 45
}

START_DIR = 'sd'
AROUND_DIR = 'ad'
NORM_DIR = 'nd'
MAXANGLE = 'ma'
MAXWIDEANGLE = 'mwa'
THUMBAXIS = 'thax'

LEFT = -1
RIGHT = 1

# Number of fast finger estimates to find out what direction is the palm
SIDE_N_ESTIM = 3

normdirrates = {}
for finger in FINGERS:
    normdirrates[finger] = {
        START_DIR: np.sin(normdirangles[finger]) ** 2,
        NORM_DIR: np.cos(normdirangles[finger]) ** 2
    }


def compute_hand_world_joints(base_hand_model, image_joints, side=RIGHT, cal=None, executor=None):
    """
    Compute the hand model in the space
    :param base_hand_model: the standard hand prototype to use for interpolation
    :param image_joints: the image points of the joints
    :param side: one of RIGHT or LEFT, the side of the hand
    :param cal: the calibration of the image joints to use
    :return: The space model of the hand
    """

    # First thing is: compute the world model correspondences of the image points
    # This way we have defined the lines that we have to interpolate
    if cal is None:
        cal = current_calibration
    world_image_info = hand_format([joint.to_camera_model(calibration=cal).as_row()
                                    for joint in raw(image_joints)])

    # for now we assume that no point is visible, so they are all directional versors.

    # the hand model is normalized with respect to the wrist
    base_hand_model = hand_format([elem - base_hand_model[WRIST][0] for elem in raw(base_hand_model)])

    # based on the palm plane, extract the transformation that maps the model to the lines
    # conserving the distances between all points.
    # this transformation provides exact correspondence between prototype and image
    tr, rotmat = get_best_first_transform(base_hand_model, world_image_info, side=side)

    # apply this first transformation to the whole model
    first_hand_model = hand_format([first_transform_point(rotmat, tr, p)
                                    for p in raw(base_hand_model)])

    # now we have to face the fingers: start getting the palm direction
    palm_base_axis = normalize(np.cross(first_hand_model[INDEX][0] - first_hand_model[WRIST][0],
                                        first_hand_model[BABY][0] - first_hand_model[WRIST][0]))

    end_model = {WRIST: first_hand_model[WRIST]}

    if executor is None:
        # if no executor pool has been provided, take care of all the fingers yourself
        for finger in (INDEX, MIDDLE, RING, BABY):
            finger_position = compute_generic_finger(first_hand_model,
                                                     palm_base_axis,
                                                     world_image_info,
                                                     finger)
            end_model[finger] = [first_hand_model[finger][0], 0, 0, 0]
            for idx in range(1, 4):
                end_model[finger][idx] = finger_position[idx - 1]

        # the thumb has special rotations that need peculiar attention:
        end_model[THUMB] = compute_thumb(first_hand_model,
                                         palm_base_axis,
                                         world_image_info)
    else:
        # if some angel provided any executor, schedule the four fingers
        futures = {}
        task = compute_generic_finger_wrap(first_hand_model,
                                           palm_base_axis,
                                           world_image_info)
        for finger in (INDEX, MIDDLE, RING, BABY):
            futures[finger] = executor.submit(task, finger)

        # and then solve the thumb problem
        end_model[THUMB] = compute_thumb(first_hand_model,
                                         palm_base_axis,
                                         world_image_info)

        # finally harvest the results
        for finger in (INDEX, MIDDLE, RING, BABY):
            end_model[finger] = [first_hand_model[finger][0], 0, 0, 0]
            for idx in range(1, 4):
                end_model[finger][idx] = futures[finger].result()[idx - 1]

    # and finally we have it
    end_check(base_hand_model, end_model)
    return end_model


def compute_generic_finger_wrap(first_hand_model, palm_base_axis, world_image_info):
    return lambda f: compute_generic_finger(first_hand_model, palm_base_axis, world_image_info, finger=f)


def compute_generic_finger(first_hand_model, palm_base_axis, world_image_info, finger):
    # the build finger utility needs complex configuration, here we build it
    conf = {NORM_DIR: palm_base_axis,
            START_DIR: normalize(first_hand_model[finger][1] - first_hand_model[finger][0])
            }
    lengths = []
    conf[AROUND_DIR] = normalize(conf[START_DIR] * normdirrates[finger][START_DIR]
                                 + conf[NORM_DIR] * normdirrates[finger][NORM_DIR])
    conf[MAXANGLE] = maxangle[finger]
    conf[MAXWIDEANGLE] = maxangle[finger] / 3
    conf[THUMBAXIS] = None
    for idx in range(3):
        lengths.append(norm(first_hand_model[finger][idx + 1] - first_hand_model[finger][idx]))
    finger_position = build_finger_num(basejoint=first_hand_model[finger][0],
                                       lengths=lengths,
                                       jointversors=world_image_info[finger][1:],
                                       config=conf)
    return finger_position


def compute_thumb(first_hand_model, palm_base_axis, world_image_info):
    conf = {NORM_DIR: palm_base_axis,
            START_DIR: normalize(first_hand_model[THUMB][1] - first_hand_model[WRIST][0])}
    end_model = [0, 0, 0, 0]
    thumbrotation = get_rotation_matrix(-conf[START_DIR], angle=np.pi / 6)
    conf[NORM_DIR] = thumbrotation @ palm_base_axis
    conf[AROUND_DIR] = normalize(conf[START_DIR] * normdirrates[THUMB][START_DIR] \
                                 + conf[NORM_DIR] * normdirrates[THUMB][NORM_DIR])
    conf[MAXANGLE] = maxangle[THUMB]
    conf[MAXWIDEANGLE] = maxangle[THUMB] / 2
    conf[THUMBAXIS] = normalize(first_hand_model[INDEX][0] - first_hand_model[WRIST][0])
    lines = [world_image_info[THUMB][1]]
    lengths = [norm(first_hand_model[WRIST][0] - first_hand_model[THUMB][1])]
    for idx in range(1, 3):
        lengths.append(norm(first_hand_model[THUMB][idx + 1] - first_hand_model[THUMB][idx]))
        lines.append(world_image_info[THUMB][idx + 1])
    thmbend = build_finger_num(basejoint=first_hand_model[WRIST][0],
                               lengths=lengths,
                               jointversors=lines,
                               config=conf)
    for idx in range(1, 4):
        end_model[idx] = thmbend[idx - 1]

    # the point zero of the thumb is not a real articulation,
    # for this reason it is artificially aligned to the rest
    alignmat = get_mapping_rot(-first_hand_model[WRIST][0] + first_hand_model[THUMB][1],
                               -first_hand_model[WRIST][0] + thmbend[0])
    thumbzero = (alignmat @ (first_hand_model[THUMB][0] - first_hand_model[WRIST][0])) \
                + first_hand_model[WRIST][0]
    end_model[0] = thumbzero

    return end_model


def end_check(model1, model2):
    """
    DEBUG UTILITY
    Check whether the two given models have any length differences
    """

    def segment_len(model, finger, idx):
        return norm(model[finger][idx] - model[finger][idx + 1])

    def inconsistency(finger, idx):
        return np.abs(segment_len(model1, finger, idx) - segment_len(model2, finger, idx))

    for finger in FINGERS:
        for idx in range(3):
            if inconsistency(finger, idx) > 1e-4:
                print("Inconsistency in %s %d by %f" % (finger, idx, inconsistency(finger, idx)))


def first_transform_point(rotmat, translation, point):
    """
    The first transformation done in the process of hand building.
    This is an affine transformation of the whole model
    """
    return (rotmat @ point) + translation


def get_rotation_angle_around_axis(axis, p1, p2):
    v1 = normalize(p1 - axis * np.dot(p1, axis))
    v2 = normalize(p2 - axis * np.dot(p2, axis))
    cross = np.cross(v1, v2)
    if np.dot(axis, cross) > 0:
        return np.arccos(np.dot(v1, v2))
    return -np.arccos(np.dot(v1, v2))


# DEBUG UTILS
global_canvas = None
global_calibration = None


def drawpnts(ptd, fill=None, cal=global_calibration):
    if cal is not None:
        ptd = [ModelPoint(pt).to_image_space(cal).as_row() for pt in ptd]
    else:
        ptd = [ModelPoint(pt).to_image_space(global_calibration).as_row() for pt in ptd]
    for pt in ptd:
        global_canvas.create_oval(pt[0] - 5,
                                  pt[1] - 5,
                                  pt[0] + 5,
                                  pt[1] + 5,
                                  fill=fill,
                                  tag="debug")


def get_best_first_transform(base_hand_model, world_image_info, side):
    base_triangle_pts = [base_hand_model[WRIST][0], base_hand_model[INDEX][0], base_hand_model[BABY][0]]
    base_triangle_lines = [world_image_info[WRIST][0], world_image_info[INDEX][0], world_image_info[BABY][0]]
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
    SIDE_N_ESTIM = 3
    avg_fingers = np.array([0., 0., 0.])
    for finger in FINGERS:
        base = first_transform_point(rotmat1, tr1, base_hand_model[finger][0])
        joints, _ = build_finger_fast(basejoint=base,
                                      lengths=[norm(base_hand_model[finger][idx + 1] - base_hand_model[finger][idx])
                                               for idx in range(SIDE_N_ESTIM)],
                                      jointversors=[world_image_info[finger][idx + 1] for idx in range(SIDE_N_ESTIM)])
        # drawpnts(joints, fill="magenta")
        avg_fingers = avg_fingers + np.average(joints)
    avg_fingers = avg_fingers / len(FINGERS)

    if np.dot(avg_fingers - model1[0], model_axis_unnorm) > 0:
        return tr1, rotmat1
    return get_transformation_from_model(model2)


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


def plane_rot_project(plane_versors, vec):
    comp1 = np.dot(plane_versors[0], vec) * plane_versors[0]
    comp2 = np.dot(plane_versors[1], vec) * plane_versors[1]
    return get_mapping_rot(vec, comp1 + comp2) @ vec


def sphere_line_intersection(center, radius, line):
    # <tv-c, tv-c> = r2
    # t2 -2<v, c>t + <c, c> - r2 = 0
    bh = - np.dot(line, center)
    delta = bh ** 2 - norm(center) + radius ** 2
    if delta < 0:
        sol = - bh * line
        return sol, sol, norm(np.cross(sol, line)) * 1e+20, True
    return (np.sqrt(delta) - bh) * line, (-bh - np.sqrt(delta)) * line, 0, False


def build_finger_num(basejoint, lengths, jointversors, config=None, cal=global_calibration):
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
        newconf[MAXWIDEANGLE] /= 10
        newconf[THUMBAXIS] = None
        return newconf

    suggest = build_finger_fast(basejoint=basejoint,
                                lengths=[lengths[0]],
                                jointversors=[jointversors[0]])[0]

    suggest.append(cone_project(axis=config[AROUND_DIR],
                                vec=suggest[0] - basejoint,
                                coneangle=config[MAXANGLE]) + basejoint)

    joint = find_best_point_in_cone(center=basejoint,
                                    radius=lengths[0],
                                    objline=jointversors[0],
                                    norm_vers=config[AROUND_DIR],
                                    tang_vers=config[NORM_DIR],
                                    normcos=np.cos(config[MAXANGLE]),
                                    planecos=np.cos(config[MAXWIDEANGLE]),
                                    suggestion=suggest)
    return [joint] + build_finger_num(joint, lengths[1:], jointversors[1:], compute_config(joint), cal=cal)


def build_finger_fast(basejoint, lengths, jointversors):
    if len(lengths) == 0:
        return [], 0

    p1, p2, loss, degen = sphere_line_intersection(center=basejoint,
                                                   radius=lengths[0],
                                                   line=jointversors[0])
    if degen:
        finger, moreloss = build_finger_fast(p1,
                                             lengths[1:],
                                             jointversors[1:])
        return [p1] + finger, loss + moreloss
    finger1, moreloss1 = build_finger_fast(p1,
                                           lengths[1:],
                                           jointversors[1:])
    if moreloss1 == 0:
        return [p1] + finger1, 0
    finger2, moreloss2 = build_finger_fast(p2,
                                           lengths[1:],
                                           jointversors[1:])
    if moreloss2 < moreloss1:
        return [p2] + finger2, moreloss2
    return [p1] + finger1, moreloss1


# testing with some fancy animations!
if __name__ == '__main__':
    import tkinter as tk
    import image_loader.hand_io as hio
    import gui.pinpointer_canvas as ppc
    import time
    import threading
    import geometry.transforms as tr
    from geometry.calibration import *
    from gui.model_drawer import ModelDrawer
    import numpy as np

    root = tk.Tk()
    frame = tk.Frame(root)
    frame.pack()
    subj_img, subj_lab = hio.load("gui/hand.mat")
    test_subject = ppc.PinpointerCanvas(frame)
    test_subject.set_bitmap(subj_img)
    test_subject.pack()
    global_canvas = test_subject
    md = ModelDrawer()
    md.set_joints(hand_format(subj_lab))
    md.set_target_area(test_subject)

    pool = fut.ThreadPoolExecutor(10)


    # here we define the hand model setup and running procedure
    # NB: not working at all.
    def loop():
        label_data = subj_lab.copy()
        resolution = subj_img.shape[0:2]
        fov = 40
        cal = calibration(intr=synth_intrinsic(resolution, (fov, fov * subj_img.shape[1] / subj_img.shape[0])))
        global global_calibration
        global_calibration = cal

        formatted_data = hand_format([ImagePoint((x * resolution[1], y * resolution[0]))
                                      for (x, y, f) in label_data])
        # compute the rotation matrix
        rotation = tr.get_rotation_matrix(axis=1, angle=np.pi / 180)

        def total():
            global hand_model
            hand_model = raw(compute_hand_world_joints(build_default_hand_model(),
                                                       formatted_data,
                                                       cal=cal,
                                                       executor=None))

        # make sure that the GUI-related load is expired before measuring performance
        # time.sleep(100)
        rep = 1
        print("Model computation %d times took %f seconds." % (rep, timeit.timeit(total, number=rep)))

        current_rotation = tr.get_rotation_matrix(axis=1, angle=0)
        time.sleep(1)
        while True:
            # rotate the 3D dataset
            center = np.average(hand_model, axis=0)
            rotated_model = [np.matmul(current_rotation, elem - center) + center for elem in hand_model]
            # project it into image space
            flat_2d = [ModelPoint(elem)
                           .to_image_space(calibration=cal, makedepth=True)
                       for elem in rotated_model]
            # normalize it before feeding to the model drawer
            # feed to model drawer
            md.set_joints(flat_2d, resolution=resolution)
            current_rotation = np.matmul(current_rotation, rotation)
            time.sleep(0.02)


    threading.Thread(target=loop).start()
    root.mainloop()
