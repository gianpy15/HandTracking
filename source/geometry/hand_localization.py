from geometry.formatting import *
import tensorflow as tf
from image_loader.hand_io import *
from geometry.calibration import *


def code_name(finger, joint=0, append=''):
    return "%s%d%s" % (finger, joint, append)


POINT_DISTANCE = 'pod'
LINE_DISTANCE = 'ld'
JOINTS_ANGLE = 'ja'

BASE = 'bs'
MID = 'md'
TOP = 'tp'
WIDE = 'wd'

ANGLE_DOMAINS = {
    WIDE: (-np.pi/180*10, np.pi/180*10),
    BASE: (-np.pi/180*20, np.pi/2),
    MID: (0, np.pi/2),
    TOP: (0, np.pi/2)
}


LOSSES = (POINT_DISTANCE,
          LINE_DISTANCE,
          JOINTS_ANGLE)


def build_default_hand_model():
    SCALE_FACTOR = 1
    img, raw_positions = load("gui/sample_hand.mat")

    raw_positions = [(x*img.shape[1], y*img.shape[0]) for (x, y, f) in raw_positions]
    cal = calibration(intr=synth_intrinsic(resolution=img.shape[0:2], fov=(50, 50*img.shape[0]/img.shape[1])))
    points = np.array([ImagePoint(elem, depth=0.2 + np.random.normal(loc=0.0, scale=1e-6)).to_camera_model(
        calibration=cal).as_row()
                       for elem in raw_positions * SCALE_FACTOR])
    return hand_format(points - np.average(points, axis=0))


class HandLocator:
    epsilon = None
    dtype = tf.float32

    def __init__(self):
        init_dict = hand_format(np.zeros(shape=(21,)))
        self.loss_weights = {}

        # Data inputs
        self.data_placeholders = {}
        self.flags_placeholders = {}

        # Model variables
        self.translation = None
        self.palm_rotations = None
        self.base_rotations_close = {}
        self.base_rotations_wide = {}
        self.mid_rotations = {}
        self.tip_rotations = {}

        # Model partial calculations
        self.base_versors = {}
        self.mid_versors = {}
        self.tip_versors = {}

        # Model partial transformations
        self.base_model = init_dict.copy()
        self.base_rotated_model = init_dict.copy()
        self.mid_rotated_model = init_dict.copy()
        self.tip_rotated_model = init_dict.copy()
        self.final_model = init_dict.copy()

        # Others
        self.feed_dict = {}
        self.losses = {}
        self.loss = None
        self.lr = 0.005
        self.active = False
        self.out_model = {}
        self.currloss = np.inf
        self.graph = None

    def setup(self, base_model=build_default_hand_model(), init=np.zeros(shape=(3,))):
        HandLocator.epsilon = tf.constant(value=1e-8, dtype=HandLocator.dtype)
        self.loss_weights = {
            POINT_DISTANCE: 1,
            LINE_DISTANCE: 1,
            JOINTS_ANGLE: 1
        }
        self.data_placeholders = {
            (WRIST, 0): tf.placeholder(dtype=HandLocator.dtype, shape=(3,), name=code_name(WRIST, append='-d'))
        }
        self.flags_placeholders = {
            (WRIST, 0): tf.placeholder(dtype=HandLocator.dtype, shape=(1,), name=code_name(WRIST, append='-f'))
        }
        for finger in FINGERS:
            for joint in range(4):
                self.data_placeholders[(finger, joint)] = tf.placeholder(dtype=HandLocator.dtype,
                                                                         shape=(3,),
                                                                         name=code_name(finger, joint, append='-d'))
                self.flags_placeholders[(finger, joint)] = tf.placeholder(dtype=HandLocator.dtype,
                                                                          shape=(1,),
                                                                          name=code_name(finger, joint, append='-f'))
        self.feed_dict = {}

        self.translation = tf.Variable(initial_value=init, dtype=HandLocator.dtype)
        self.palm_rotations = tf.Variable(initial_value=np.zeros(shape=(3,)), dtype=HandLocator.dtype)
        for finger in FINGERS:
            self.base_rotations_close[finger] = tf.Variable(initial_value=np.zeros(shape=(1,)), dtype=HandLocator.dtype)
            self.base_rotations_wide[finger] = tf.Variable(initial_value=np.zeros(shape=(1,)), dtype=HandLocator.dtype)
            self.mid_rotations[finger] = tf.Variable(initial_value=np.zeros(shape=(1,)), dtype=HandLocator.dtype)
            self.tip_rotations[finger] = tf.Variable(initial_value=np.zeros(shape=(1,)), dtype=HandLocator.dtype)

        self.base_model = hand_format([tf.constant(value=joint, dtype=HandLocator.dtype) for joint in raw(base_model)])

        for finger in FINGERS:
            # First rotate all fingers by the base angles
            base_versor = HandLocator.normalized_internal_cross(self.base_model[WRIST][0],
                                                                self.base_model[finger][0],
                                                                self.base_model[finger][1])
            finger_directed_versor = HandLocator.normalized_diff(self.base_model[finger][1],
                                                                 self.base_model[finger][0])
            basechange_mat = HandLocator.get_base_change_mat(base_versor, finger_directed_versor)
            self.base_rotated_model[WRIST][0] = self.base_model[WRIST][0]
            self.base_rotated_model[finger][0] = self.base_model[finger][0]
            for idx in range(1, 4):
                self.base_rotated_model[finger][idx] = \
                    tf.matmul(tf.transpose(basechange_mat),
                              tf.matmul(HandLocator.get_rotation_matrix(axis=2,
                                                                        angle=self.base_rotations_wide[finger]),
                                        tf.matmul(HandLocator.get_rotation_matrix(axis=0,
                                                                                  angle=self.base_rotations_close[
                                                                                      finger]),
                                                  tf.matmul(basechange_mat,
                                                            HandLocator.ascol(self.base_model[finger][idx] -
                                                                              self.base_model[finger][0]))))) + \
                    HandLocator.ascol(self.base_model[finger][0])
                self.base_rotated_model[finger][idx] = HandLocator.asrow(self.base_rotated_model[finger][idx])

            # Then rotate mids and tips by the mid angles
            base_versor = HandLocator.normalized_internal_cross(self.base_rotated_model[finger][0],
                                                                self.base_rotated_model[finger][1],
                                                                self.base_rotated_model[finger][2])
            finger_directed_versor = HandLocator.normalized_diff(self.base_rotated_model[finger][2],
                                                                 self.base_rotated_model[finger][1])
            basechange_mat = HandLocator.get_base_change_mat(base_versor, finger_directed_versor)
            self.mid_rotated_model[WRIST][0] = self.base_rotated_model[WRIST][0]
            for idx in range(2):
                self.mid_rotated_model[finger][idx] = self.base_rotated_model[finger][idx]
            for idx in range(2, 4):
                self.mid_rotated_model[finger][idx] = \
                    tf.matmul(tf.transpose(basechange_mat),
                              tf.matmul(HandLocator.get_rotation_matrix(axis=0,
                                                                        angle=self.mid_rotations[finger]),
                                        tf.matmul(basechange_mat,
                                                  HandLocator.ascol(self.base_rotated_model[finger][idx] -
                                                                    self.base_rotated_model[finger][1])))) + \
                    HandLocator.ascol(self.base_rotated_model[finger][1])
                self.mid_rotated_model[finger][idx] = HandLocator.asrow(self.mid_rotated_model[finger][idx])
            # Then rotate tips by the tip angles
            base_versor = HandLocator.normalized_internal_cross(self.mid_rotated_model[finger][1],
                                                                self.mid_rotated_model[finger][2],
                                                                self.mid_rotated_model[finger][3])
            finger_directed_versor = HandLocator.normalized_diff(self.mid_rotated_model[finger][3],
                                                                 self.mid_rotated_model[finger][2])
            basechange_mat = HandLocator.get_base_change_mat(base_versor, finger_directed_versor)

            self.tip_rotated_model[WRIST][0] = self.mid_rotated_model[WRIST][0]
            for idx in range(3):
                self.tip_rotated_model[finger][idx] = self.mid_rotated_model[finger][idx]
            self.tip_rotated_model[finger][3] = \
                tf.matmul(tf.transpose(basechange_mat),
                          tf.matmul(HandLocator.get_rotation_matrix(axis=0,
                                                                    angle=self.tip_rotations[finger]),
                                    tf.matmul(basechange_mat,
                                              HandLocator.ascol(self.mid_rotated_model[finger][3] -
                                                                self.mid_rotated_model[finger][2])))) + \
                HandLocator.ascol(self.mid_rotated_model[finger][2])
            self.tip_rotated_model[finger][3] = HandLocator.asrow(self.tip_rotated_model[finger][3])
        # Finally we need to translate and rotate all this mess
        rotx, roty, rotz = tf.split(self.palm_rotations, 3)
        self.final_model = hand_format([HandLocator.asrow(tf.matmul(HandLocator.get_rotation_matrix(axis=2, angle=rotz),
                                                                    tf.matmul(HandLocator.get_rotation_matrix(axis=1,
                                                                                                              angle=roty),
                                                                              tf.matmul(
                                                                                  HandLocator.get_rotation_matrix(
                                                                                      axis=0, angle=rotx),
                                                                                  HandLocator.ascol(elem)))) +
                                                          HandLocator.ascol(self.translation))
                                        for elem in raw(self.tip_rotated_model)])

        self.losses = {
            POINT_DISTANCE: self.point_distance_losses(),
            LINE_DISTANCE: self.line_distance_losses(),
            JOINTS_ANGLE: self.joints_angle_losses()
        }

        self.loss = tf.reduce_sum([self.losses[l] * self.loss_weights[l] for l in LOSSES])

        self.active = False
        self.out_model = hand_format(raw(base_model) + init)
        self.currloss = np.inf
        self.graph = tf.get_default_graph()

    def start(self):
        self.active = True
        with self.graph.as_default():
            opt = tf.train.AdamOptimizer(self.lr)
            train = opt.minimize(self.loss)
            init = tf.global_variables_initializer()
            with tf.Session() as session:
                session.run(init)
                while self.active:
                    print(session.run(self.losses, feed_dict=self.feed_dict))
                    _, self.currloss, hand_model = session.run([train, self.loss, self.final_model],
                                                               feed_dict=self.feed_dict)
                    self.out_model = hand_model.copy()

    def input_image_data(self, image_points_dict):
        for key in image_points_dict.keys():
            for joint in range(len(image_points_dict[key])):
                self.feed_dict[self.data_placeholders[(key, joint)]] = \
                    image_points_dict[key][joint].to_camera_model().as_row()
                self.feed_dict[self.flags_placeholders[(key, joint)]] = \
                    (0,) if image_points_dict[key][joint].visible else (1,)

    # ################################INIT-MADE LOSSES################################
    # POD - POINT_DISTANCE
    def point_distance_losses(self):
        losses = [HandLocator.fixed_point_loss(self.final_model[WRIST][0], self.data_placeholders[(WRIST, 0)])]
        losses[-1] *= tf.constant(value=1, dtype=HandLocator.dtype) - self.flags_placeholders[(WRIST, 0)]
        for finger in FINGERS:
            for joint in range(4):
                losses.append(HandLocator.fixed_point_loss(self.final_model[finger][joint],
                                                           self.data_placeholders[(finger, joint)]))
                losses[-1] *= tf.constant(value=1, dtype=HandLocator.dtype) - self.flags_placeholders[(finger, joint)]
        return tf.reduce_sum(losses)

    # LD - LINE_DISTANCE
    def line_distance_losses(self):
        losses = [HandLocator.linewise_loss(self.final_model[WRIST][0], self.data_placeholders[(WRIST, 0)])]
        losses[-1] *= self.flags_placeholders[(WRIST, 0)]
        for finger in FINGERS:
            for joint in range(4):
                losses.append(HandLocator.linewise_loss(self.final_model[finger][joint],
                                                        self.data_placeholders[(finger, joint)]))
                losses[-1] *= self.flags_placeholders[(finger, joint)]
        return tf.reduce_sum(losses)

    def joints_angle_losses(self):
        losses = []
        for finger in FINGERS:
            losses.append(HandLocator.interval_loss(self.base_rotations_close[finger], ANGLE_DOMAINS[BASE]))
            losses.append(HandLocator.interval_loss(self.base_rotations_wide[finger], ANGLE_DOMAINS[WIDE]))
            losses.append(HandLocator.interval_loss(self.mid_rotations[finger], ANGLE_DOMAINS[MID]))
            losses.append(HandLocator.interval_loss(self.tip_rotations[finger], ANGLE_DOMAINS[TOP]))
        return tf.reduce_sum(losses)

    @staticmethod
    def interval_loss(value, interval):
        tr = (interval[1]+interval[0])/2
        dw = (interval[1]-interval[0])/2
        return tf.exp(180 / np.pi * tf.nn.relu(tf.abs(value-tr)-dw)) - 1

    @staticmethod
    def fixed_point_loss(var, ph):
        return tf.norm(tf.subtract(var, ph) + HandLocator.epsilon)

    @staticmethod
    def linewise_loss(var, ph):
        return tf.norm(HandLocator.cross_product(var, ph) + HandLocator.epsilon)

    @staticmethod
    def inner_cosine(v1, v2, v3):
        diff1 = HandLocator.normalized_diff(v3, v2)
        diff2 = HandLocator.normalized_diff(v1, v2)
        return HandLocator.unnormalized_cosine(diff1, diff2)

    @staticmethod
    def normalized_diff(p1, p2):
        diff = tf.subtract(p1, p2)
        return diff / (HandLocator.epsilon + tf.norm(diff))

    @staticmethod
    def unnormalized_cosine(p1, p2):
        return tf.tensordot(p1, p2, axes=[[0], [0]])

    @staticmethod
    def plane_versor(v1, v2, v3):
        return HandLocator.cross_product(HandLocator.normalized_diff(v2, v1),
                                         HandLocator.normalized_diff(v3, v1))

    @staticmethod
    def normalized_internal_cross(p1, p2, p3):
        diff1 = tf.subtract(p2, p1)
        diff2 = tf.subtract(p3, p2)
        cross = HandLocator.cross_product(diff1, diff2)
        return cross / (HandLocator.epsilon + tf.norm(cross))

    @staticmethod
    def cross_product(v1, v2):
        x1, y1, z1 = tf.unstack(v1)
        x2, y2, z2 = tf.unstack(v2)
        x3 = y1 * z2 - y2 * z1
        y3 = x2 * z1 - z2 * x1
        z3 = x1 * y2 - x2 * y1
        return tf.stack((x3, y3, z3))

    @staticmethod
    def get_base_change_mat(v1, v2):
        v3 = HandLocator.cross_product(v1, v2)
        return tf.stack((v1, v2, v3))

    @staticmethod
    def get_rotation_matrix(axis, angle):
        s = tf.sin(angle)
        c = tf.cos(angle)
        if axis == 0:
            # around x axis:
            return tf.stack((tf.constant(value=[1, 0, 0], dtype=HandLocator.dtype),
                             tf.concat((tf.zeros(shape=(1,), dtype=HandLocator.dtype), c, -s), 0),
                             tf.concat((tf.zeros(shape=(1,), dtype=HandLocator.dtype), s, c), 0)))
        elif axis == 1:
            # around y axis:
            return tf.stack((tf.concat((c, tf.zeros(shape=(1,), dtype=HandLocator.dtype), -s), 0),
                             tf.constant(value=[0, 1, 0], dtype=HandLocator.dtype),
                             tf.concat((s, tf.zeros(shape=(1,), dtype=HandLocator.dtype), c), 0)))
        else:
            # around z axis:
            return tf.stack((tf.concat((c, -s, tf.zeros(shape=(1,), dtype=HandLocator.dtype)), 0),
                             tf.concat((s, c, tf.zeros(shape=(1,), dtype=HandLocator.dtype)), 0),
                             tf.constant(value=[0, 0, 1], dtype=HandLocator.dtype)))

    @staticmethod
    def ascol(v):
        return tf.expand_dims(v, axis=1)

    @staticmethod
    def asrow(v):
        return tf.unstack(v, axis=1)[0]


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
    md = ModelDrawer()
    md.set_joints(hand_format(helper_hand_lab))
    md.set_target_area(test_subject)
    hand_locator = HandLocator()


    # here we define the hand model setup and running procedure
    # NB: not working at all.
    def loop():
        label_data = helper_hand_lab.copy()
        resolution = helper_hand_img.shape[0:2]
        set_current_calib(calibration(intr=synth_intrinsic(resolution, (50, 50))))

        formatted_data = hand_format([ImagePoint((x * resolution[1], y * resolution[0]))
                                      for (x, y, f) in label_data])
        # compute the rotation matrix
        rotation = tr.get_rotation_matrix(axis=1, angle=np.pi / 180 * 3)
        # build the optimizer
        initval = np.average([point.to_camera_model().as_row()
                              for point in raw(formatted_data)],
                             axis=0)
        hand_locator.setup(init=initval)

        def manager_setup_and_run():
            hand_locator.start()

        threading.Thread(target=manager_setup_and_run).start()
        current_rotation = tr.get_rotation_matrix(axis=1, angle=0)
        hand_locator.input_image_data(formatted_data)
        while True:
            hand_model = raw(hand_locator.out_model.copy())
            # rotate the 3D dataset
            center = np.average(hand_model, axis=0)
            hand_model = [np.matmul(current_rotation, elem - center) + center for elem in hand_model]
            # project it into image space
            flat_2d = [ModelPoint(elem)
                           .to_image_space()
                           .as_row()
                       for elem in hand_model]
            # normalize it before feeding to the model drawer
            flat_2d_norm = [(x / resolution[1], y / resolution[0]) for (x, y) in flat_2d]
            # feed to model drawer
            md.set_joints(hand_format(flat_2d_norm))
            current_rotation = np.matmul(current_rotation, rotation)
            time.sleep(0.04)


    threading.Thread(target=loop).start()
    root.mainloop()
