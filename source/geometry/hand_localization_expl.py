from geometry.calibration import *
from geometry.formatting import *
from geometry.transforms import *
from numpy.linalg import norm
from geometry.hand_localization import build_default_hand_model
import timeit

maxangle = {
    THUMB: np.pi / 180 * 30,
    INDEX: np.pi / 180 * 55,
    MIDDLE: np.pi / 180 * 50,
    RING: np.pi / 180 * 46,
    BABY: np.pi / 180 * 47
}

normdirangles = {
    THUMB: np.pi / 180 * 25,
    INDEX: np.pi / 180 * 45,
    MIDDLE: np.pi / 180 * 45,
    RING: np.pi / 180 * 45,
    BABY: np.pi / 180 * 45
}

START_DIR = 'sd'
AROUND_DIR = 'ad'
NORM_DIR = 'nd'
MAXANGLE = 'ma'

normdirrates = {}
for finger in FINGERS:
    normdirrates[finger] = {
        START_DIR: np.sin(normdirangles[finger]) ** 2,
        NORM_DIR: np.cos(normdirangles[finger]) ** 2
    }


def compute_hand_world_joints(base_hand_model, image_joints, cal=None):
    if cal is None:
        cal = current_calibration
    world_image_info = hand_format([joint.to_camera_model(calibration=cal).as_row()
                                    for joint in raw(image_joints)])

    # for now we assume that no point is visible, so they are all directional versors.
    base_hand_model = hand_format([elem - base_hand_model[WRIST][0] for elem in raw(base_hand_model)])
    rotmat, tr = get_best_first_transform(base_hand_model, world_image_info, cal)

    first_hand_model = hand_format([first_transform_point(rotmat, tr, p)
                                    for p in raw(base_hand_model)])

    palm_base_axis = normalize(cross_product(first_hand_model[INDEX][0] - first_hand_model[WRIST][0],
                                             first_hand_model[BABY][0] - first_hand_model[WRIST][0]))

    # drawpnt(raw(first_hand_model), cal=calibration)

    conf = {NORM_DIR: palm_base_axis}

    end_model = {WRIST: first_hand_model[WRIST]}
    for finger in (INDEX, MIDDLE, RING, BABY):
        conf[START_DIR] = normalize(first_hand_model[finger][1] - first_hand_model[finger][0])
        conf[AROUND_DIR] = normalize(conf[START_DIR]*normdirrates[finger][START_DIR]
                                     + conf[NORM_DIR]*normdirrates[finger][NORM_DIR])
        conf[MAXANGLE] = maxangle[finger]
        end_model[finger] = [first_hand_model[finger][0], 0, 0, 0]
        lengths = []
        for idx in range(3):
            lengths.append(norm(first_hand_model[finger][idx + 1] - first_hand_model[finger][idx]))
        finger_position, loss = build_finger(basejoint=first_hand_model[finger][0],
                                             lengths=lengths,
                                             jointversors=world_image_info[finger][1:],
                                             config=conf,
                                             calib=cal)
        for idx in range(1, 4):
            end_model[finger][idx] = finger_position[idx - 1]
        # print("Finger: %s\tLoss: %f" % (finger, loss))

    end_model[THUMB] = [0, 0, 0, 0]
    conf[START_DIR] = normalize(first_hand_model[THUMB][1] - first_hand_model[WRIST][0])
    conf[NORM_DIR] = (cross_product(palm_base_axis, conf[START_DIR]) + palm_base_axis) / np.sqrt(2)
    conf[AROUND_DIR] = normalize(conf[START_DIR]*normdirrates[THUMB][START_DIR] \
                                  + conf[NORM_DIR]*normdirrates[THUMB][NORM_DIR])
    conf[MAXANGLE] = maxangle[THUMB]
    lines = [world_image_info[THUMB][1]]
    lengths = [norm(first_hand_model[WRIST][0] - first_hand_model[THUMB][1])]
    for idx in range(1, 3):
        lengths.append(norm(first_hand_model[THUMB][idx + 1] - first_hand_model[THUMB][idx]))
        lines.append(world_image_info[THUMB][idx + 1])
    thmbend, loss = build_finger(basejoint=first_hand_model[WRIST][0],
                                 lengths=lengths,
                                 jointversors=lines,
                                 config=conf,
                                 calib=cal)
    for idx in range(1, 4):
        end_model[THUMB][idx] = thmbend[idx - 1]

    alignmat = get_mapping_rot(-first_hand_model[WRIST][0] + first_hand_model[THUMB][1],
                               -end_model[WRIST][0] + thmbend[0])
    thumbzero = column_matmul(alignmat,
                              first_hand_model[THUMB][0] - first_hand_model[WRIST][0]) \
                + first_hand_model[WRIST][0]
    end_model[THUMB][0] = thumbzero
    # print("Finger: %s\tLoss: %f" % (THUMB, loss))

    end_check(base_hand_model, end_model)

    # drawpnts(raw(world_image_info)*5000, cal=cal)

    return end_model


def end_check(model1, model2):
    def segment_len(model, finger, idx):
        return norm(model[finger][idx] - model[finger][idx + 1])

    def inconsistency(finger, idx):
        return np.abs(segment_len(model1, finger, idx) - segment_len(model2, finger, idx))

    for finger in FINGERS:
        for idx in range(3):
            if inconsistency(finger, idx) > 1e-4:
                print("Inconsistency in %s %d by %f" % (finger, idx, inconsistency(finger, idx)))


def first_transform_point(rotmat, translation, point):
    return column_matmul(rotmat, point) + translation


def get_rotation_angle_around_axis(axis, p1, p2):
    v1 = normalize(p1 - axis * np.dot(p1, axis))
    v2 = normalize(p2 - axis * np.dot(p2, axis))
    cross = cross_product(v1, v2)
    if np.dot(axis, cross) > 0:
        return np.arccos(np.dot(v1, v2))
    return -np.arccos(np.dot(v1, v2))


def column_matmul(m, v):
    res = np.matmul(m, np.expand_dims(v, 1))
    return np.reshape(res, (3,))


# DEBUG UTILS
global_canvas = None
global_calibration = None


def drawpnts(ptd, fill=None, cal=None):
    if cal is not None:
        ptd = [ModelPoint(pt).to_image_space(cal).as_row() for pt in ptd]
    for pt in ptd:
        global_canvas.create_oval(pt[0] - 5,
                                  pt[1] - 5,
                                  pt[0] + 5,
                                  pt[1] + 5,
                                  fill=fill)


def get_best_first_transform(base_hand_model, world_image_info, cal):
    base_triangle_pts = [base_hand_model[WRIST][0], base_hand_model[INDEX][0], base_hand_model[BABY][0]]
    base_triangle_lines = [world_image_info[WRIST][0], world_image_info[INDEX][0], world_image_info[BABY][0]]
    model1, model2 = get_points_projection_to_lines_pair(base_triangle_pts, base_triangle_lines)

    # drawpnt(model2, cal=cal)
    # drawpnt(model1, cal=cal)

    tr1 = model1[0]
    tr2 = model2[0]

    # drawpnt(base_triangle_pts+tr1, cal=cal, fill="light blue")

    rotmat1 = get_mapping_rot(base_triangle_pts[1], model1[1] - model1[0])
    rotmat2 = get_mapping_rot(base_triangle_pts[1], model2[1] - model2[0])

    # drawpnt([first_transform_point(rotmat1, tr1, pt) for pt in base_triangle_pts], cal=cal)
    # drawpnt([first_transform_point(rotmat2, tr2, pt) for pt in base_triangle_pts], cal=cal)

    axis1 = normalize(model1[1] - model1[0])
    axis2 = normalize(model2[1] - model2[0])

    angle1 = get_rotation_angle_around_axis(axis1,
                                            column_matmul(rotmat1,
                                                          base_triangle_pts[2]),
                                            model1[2] - model1[0])
    angle2 = get_rotation_angle_around_axis(axis2,
                                            column_matmul(rotmat2,
                                                          base_triangle_pts[2]),
                                            model2[2] - model2[0])

    rotmat1 = np.matmul(get_rotation_matrix(axis1, angle1), rotmat1)
    rotmat2 = np.matmul(get_rotation_matrix(axis2, angle2), rotmat2)

    # TODO
    # model1 and model2 represent left and right hand hypotheses
    # try to find a way to distinguish them
    loss1 = norm(first_transform_point(rotmat1, tr1, base_triangle_pts[2]) - model1[2])
    loss2 = norm(first_transform_point(rotmat2, tr2, base_triangle_pts[2]) - model2[2])

    if loss1 > loss2:
        return rotmat2, tr2
    return rotmat1, tr1


def cone_project(axis, vec, coneangle):
    """
    Compute the projection of vector vec with respect to the cone having axis as base axis
    and with aperture coneangle
    """
    rotax = normalize(cross_product(vec, axis))
    rotangle = np.arccos(np.dot(axis, vec) / norm(vec)) - coneangle
    rotmat = get_rotation_matrix(rotax, rotangle)
    return column_matmul(rotmat, vec)


def plane_rot_project(plane_versors, vec):
    comp1 = np.dot(plane_versors[0], vec) * plane_versors[0]
    comp2 = np.dot(plane_versors[1], vec) * plane_versors[1]
    rotmat = get_mapping_rot(vec, comp1 + comp2)
    return column_matmul(rotmat, vec)


def sphere_line_intersection(center, radius, line, calib):
    # <tv-c, tv-c> = r2
    # t2 -2<v, c>t + <c, c> - r2 = 0
    drawpnts([line], cal=calib, fill='dark blue')
    bh = - np.dot(line, center)
    delta = bh ** 2 - norm(center) + radius ** 2
    if delta < 0:
        sol = - bh * line
        return sol, sol, norm(cross_product(sol, line)) * 1e+20, True
    return (np.sqrt(delta) - bh) * line, (-bh - np.sqrt(delta)) * line, 0, False


def build_finger(basejoint, lengths, jointversors, config=None, forceplanar=False, forcelengths=True, calib=None):
    if len(lengths) == 0:
        return [], 0

    def check_and_correct_solution(proposed_sol):
        if config is None:
            return proposed_sol, 0
        if forceplanar:
            newsol = plane_rot_project((config[START_DIR], config[NORM_DIR]), proposed_sol - basejoint) + basejoint
        else:
            newsol = proposed_sol
        if np.dot(newsol, config[AROUND_DIR]) / norm(newsol) >= np.cos(config[MAXANGLE]):
            return newsol, 0
        newsol = cone_project(config[AROUND_DIR], newsol - basejoint, config[MAXANGLE]) + basejoint
        correctionloss = norm(proposed_sol - newsol) * 1e+20
        return newsol, correctionloss

    def compute_config(proposed_sol):
        if config is None:
            return None
        rotmat = get_mapping_rot(config[START_DIR], proposed_sol - basejoint)
        newconf = config.copy()
        newconf[START_DIR] = column_matmul(rotmat, config[START_DIR])
        newconf[AROUND_DIR] = column_matmul(rotmat, config[AROUND_DIR])
        newconf[NORM_DIR] = column_matmul(rotmat, config[NORM_DIR])
        print(np.dot(newconf[NORM_DIR], newconf[START_DIR]))

        return newconf

    p1, p2, loss, degen = sphere_line_intersection(center=basejoint,
                                                   radius=lengths[0],
                                                   line=jointversors[0],
                                                   calib=calib)
    drawpnts([p1, p2], cal=calib, fill='magenta')
    if degen:
        if forcelengths:
            # print("one end out: %f" % loss)
            ret = normalize(p1 - basejoint)
            ret = ret * lengths[0] + basejoint
        else:
            ret = p1
        ret, correctionloss = check_and_correct_solution(ret)
        finger, moreloss = build_finger(ret,
                                        lengths[1:],
                                        jointversors[1:],
                                        compute_config(ret),
                                        forceplanar=False,
                                        forcelengths=forcelengths,
                                        calib=calib)
        return [ret] + finger, loss + moreloss + correctionloss
    p1, corr1 = check_and_correct_solution(p1)
    finger1, moreloss1 = build_finger(p1,
                                      lengths[1:],
                                      jointversors[1:],
                                      compute_config(p1),
                                      forceplanar=False,
                                      forcelengths=forcelengths,
                                      calib=calib)
    if moreloss1 + corr1 == 0:
        return [p1] + finger1, 0
    p2, corr2 = check_and_correct_solution(p2)
    finger2, moreloss2 = build_finger(p2,
                                      lengths[1:],
                                      jointversors[1:],
                                      compute_config(p2),
                                      forceplanar=False,
                                      forcelengths=forcelengths,
                                      calib=calib)
    if moreloss2 + corr2 < moreloss1 + corr1:
        return [p2] + finger2, moreloss2 + corr2
    return [p1] + finger1, moreloss1 + corr1


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
    helper_hand_img, helper_hand_lab = hio.load("gui/hand.mat")
    test_subject = ppc.PinpointerCanvas(frame)
    test_subject.set_bitmap(helper_hand_img)
    test_subject.pack()
    global_canvas = test_subject
    md = ModelDrawer()
    md.set_joints(hand_format(helper_hand_lab))
    md.set_target_area(test_subject)


    # here we define the hand model setup and running procedure
    # NB: not working at all.
    def loop():
        label_data = helper_hand_lab.copy()
        resolution = helper_hand_img.shape[0:2]
        cal = calibration(intr=synth_intrinsic(resolution, (50, 50)))
        global_calibration = cal

        formatted_data = hand_format([ImagePoint((x * resolution[1], y * resolution[0]))
                                      for (x, y, f) in label_data])
        # compute the rotation matrix
        rotation = tr.get_rotation_matrix(axis=1, angle=np.pi / 180)

        def total():
            global hand_model
            hand_model = raw(compute_hand_world_joints(build_default_hand_model(),
                                                       formatted_data, cal=cal))

        # make sure that the GUI-related load is expired before measuring performance
        # time.sleep(1)
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
                       for elem in hand_model]
            # normalize it before feeding to the model drawer
            # feed to model drawer
            md.set_joints(flat_2d, resolution=resolution)
            current_rotation = np.matmul(current_rotation, rotation)
            time.sleep(0.02)


    threading.Thread(target=loop).start()
    root.mainloop()
