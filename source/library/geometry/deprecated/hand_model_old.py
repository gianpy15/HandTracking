from library.geometry.formatting import *
import tensorflow as tf


def code_name(finger, joint=0, append=''):
    return "%s%d%s" % (finger, joint, append)


POINT_DISTANCE = 'pod'
LINE_DISTANCE = 'ld'
PLANE_DISTANCE = 'pld'
FINGER_COSINE = 'fg'
INTERPLANE_ANGLE = 'ipa'
INTERJOINT_PROPORTION = 'ijp'

LOSSES = (POINT_DISTANCE,
          LINE_DISTANCE,
          PLANE_DISTANCE,
          FINGER_COSINE,
          INTERPLANE_ANGLE,
          INTERJOINT_PROPORTION)

default_interjoint_proportions = {
    ((WRIST, 0), (THUMB, 0)): 3.,
    ((WRIST, 0), (INDEX, 0)): 9.,
    ((WRIST, 0), (MIDDLE, 0)): 9.,
    ((WRIST, 0), (RING, 0)): 8.5,
    ((WRIST, 0), (BABY, 0)): 8.5,
    ((THUMB, 0), (INDEX, 0)): 7.,
    ((INDEX, 0), (MIDDLE, 0)): 2.,
    ((MIDDLE, 0), (RING, 0)): 2.,
    ((RING, 0), (BABY, 0)): 2.5,
    ((THUMB, 0), (THUMB, 1)): 4.,
    ((THUMB, 1), (THUMB, 2)): 3.,
    ((THUMB, 2), (THUMB, 3)): 3.,
    ((INDEX, 0), (INDEX, 1)): 3.,
    ((INDEX, 1), (INDEX, 2)): 2.,
    ((INDEX, 2), (INDEX, 3)): 2.5,
    ((MIDDLE, 0), (MIDDLE, 1)): 3.75,
    ((MIDDLE, 1), (MIDDLE, 2)): 2.5,
    ((MIDDLE, 2), (MIDDLE, 3)): 2.75,
    ((RING, 0), (RING, 1)): 3.25,
    ((RING, 1), (RING, 2)): 2.25,
    ((RING, 2), (RING, 3)): 2.5,
    ((BABY, 0), (BABY, 1)): 2.75,
    ((BABY, 1), (BABY, 2)): 1.5,
    ((BABY, 2), (BABY, 3)): 2.5
}


class HandModel:
    epsilon = None

    def __init__(self):
        self.loss_weights = {}
        self.data_placeholders = {}
        self.flags_placeholders = {}
        self.vars = {}
        self.common_measure_var = None
        self.interjoint_proportions = None
        self.feed_dict = {}
        self.versors = {}
        self.losses = {}
        self.loss = None
        self.lr = None
        self.active = False
        self.out_model = {}
        self.currloss = np.inf
        self.graph = None

    def setup(self, init_dict):
        HandModel.epsilon = tf.constant(value=1e-8, dtype=tf.float32)
        self.loss_weights = {
            POINT_DISTANCE: 0,
            LINE_DISTANCE: 0,
            PLANE_DISTANCE: 1e+3,
            FINGER_COSINE: 1e+5,
            INTERPLANE_ANGLE: 10,
            INTERJOINT_PROPORTION: 1e+2
        }
        self.data_placeholders = {
            (WRIST, 0): tf.placeholder(dtype=tf.float32, shape=(3,), name=code_name(WRIST, append='-d'))
        }
        self.flags_placeholders = {
            (WRIST, 0): tf.placeholder(dtype=tf.float32, shape=(1,), name=code_name(WRIST, append='-f'))
        }
        self.vars = {
            (WRIST, 0): self.data_placeholders[(WRIST, 0)] * tf.Variable(dtype=tf.float32,
                                                                         # initial_value=init_dict[WRIST][0].as_row(),
                                                                         initial_value=1.0,
                                                                         name=code_name(WRIST, append='-v'))
        }
        for finger in FINGERS:
            for joint in range(4):
                self.data_placeholders[(finger, joint)] = tf.placeholder(dtype=tf.float32,
                                                                         shape=(3,),
                                                                         name=code_name(finger, joint, append='-d'))
                self.flags_placeholders[(finger, joint)] = tf.placeholder(dtype=tf.float32,
                                                                          shape=(1,),
                                                                          name=code_name(finger, joint, append='-f'))
                self.vars[(finger, joint)] = self.data_placeholders[(finger, joint)] * tf.Variable(dtype=tf.float32,
                                                                                                   # initial_value=init_dict[finger][joint].as_row(),
                                                                                                   initial_value=1.0,
                                                                                                   name=code_name(
                                                                                                       finger,
                                                                                                       append='-v'))
        self.common_measure_var = tf.Variable(dtype=tf.float32, initial_value=0.1, name='ref_measure')
        self.interjoint_proportions = default_interjoint_proportions
        self.feed_dict = {}
        self.versors = {
            WRIST: HandModel.plane_versor(self.vars[(WRIST, 0)],
                                          self.vars[(INDEX, 0)],
                                          self.vars[(BABY, 0)]),
        }
        for finger in FINGERS:
            self.versors[finger] = HandModel.plane_versor(self.vars[(finger, 0)],
                                                          self.vars[(finger, 1)],
                                                          self.vars[(finger, 2)])

        self.losses = {
            POINT_DISTANCE: self.point_distance_losses(),
            LINE_DISTANCE: self.line_distance_losses(),
            PLANE_DISTANCE: self.plane_distance_losses(),
            FINGER_COSINE: self.finger_cosine_losses(),
            INTERPLANE_ANGLE: self.inter_plane_angle_losses(),
            INTERJOINT_PROPORTION: self.inter_joint_proportion_losses()
        }

        self.loss = tf.reduce_sum([self.losses[l] * self.loss_weights[l] for l in LOSSES])

        self.lr = 0.001
        self.active = False
        self.out_model = hand_format([j.as_row() for j in raw(init_dict)])
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
                    _, self.currloss, varsdict = session.run([train, self.loss, self.vars], feed_dict=self.feed_dict)
                    self.set_current_estimate(varsdict)

    def set_current_estimate(self, varsdict):
        for (finger, joint) in varsdict.keys():
            self.out_model[finger][joint] = varsdict[(finger, joint)]

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
        losses = [HandModel.fixed_point_loss(self.vars[(WRIST, 0)], self.data_placeholders[(WRIST, 0)])]
        losses[-1] *= tf.constant(value=1, dtype=tf.float32) - self.flags_placeholders[(WRIST, 0)]
        for finger in FINGERS:
            for joint in range(4):
                losses.append(HandModel.fixed_point_loss(self.vars[(finger, joint)],
                                                         self.data_placeholders[(finger, joint)]))
                losses[-1] *= tf.constant(value=1, dtype=tf.float32) - self.flags_placeholders[(finger, joint)]
        return tf.reduce_sum(losses)

    # LD - LINE_DISTANCE
    def line_distance_losses(self):
        losses = [HandModel.linewise_loss(self.vars[(WRIST, 0)], self.data_placeholders[(WRIST, 0)])]
        losses[-1] *= self.flags_placeholders[(WRIST, 0)]
        for finger in FINGERS:
            for joint in range(4):
                losses.append(HandModel.linewise_loss(self.vars[(finger, joint)],
                                                      self.data_placeholders[(finger, joint)]))
                losses[-1] *= self.flags_placeholders[(finger, joint)]
        return tf.reduce_sum(losses)

    # PLD - PLANE_DISTANCE
    def plane_distance_losses(self):
        losses = [HandModel.point_to_plane_loss(self.versors[WRIST],
                                                -tf.tensordot(self.versors[WRIST],
                                                              self.vars[(WRIST, 0)],
                                                              axes=[[0], [0]]),
                                                [self.vars[(MIDDLE, 0)],
                                                 self.vars[(RING, 0)]]
                                                )]
        for finger in FINGERS:
            losses.append(HandModel.point_to_plane_loss(self.versors[finger],
                                                        -tf.tensordot(self.versors[finger],
                                                                      self.vars[(finger, 0)],
                                                                      axes=[[0], [0]]),
                                                        [self.vars[(finger, 3)]]
                                                        ))
        return tf.reduce_sum(losses)

    # FG - FINGER_COSINE
    def finger_cosine_losses(self):
        losses = []
        for finger in FINGERS:
            base_direction = HandModel.normalized_diff(self.vars[(finger, 1)],
                                                       self.vars[(finger, 0)])
            losses.append(tf.nn.relu(-HandModel.unnormalized_cosine(base_direction,
                                                                    self.versors[WRIST])))
        return tf.reduce_sum(losses)

    # IPA - INTER_PLANE_ANGLE
    def inter_plane_angle_losses(self):
        losses = []
        for finger in FINGERS:
            losses.append(tf.norm(tf.tensordot(self.versors[WRIST],
                                               self.versors[finger],
                                               axes=[[0], [0]])))
        return tf.reduce_sum(losses)

    # IJP - INTER_JOINT_PROPORTION
    def inter_joint_proportion_losses(self):
        losses = []
        for (j1, j2) in default_interjoint_proportions.keys():
            losses.append(tf.norm(HandModel.fixed_point_loss(self.vars[j1], self.vars[j2])
                                  - tf.multiply(tf.constant(value=self.interjoint_proportions[(j1, j2)],
                                                            dtype=tf.float32),
                                                self.common_measure_var)))
        return tf.reduce_sum(losses)

    @staticmethod
    def fixed_point_loss(var, ph):
        return tf.norm(tf.subtract(var, ph) + HandModel.epsilon)

    @staticmethod
    def linewise_loss(var, ph):
        return tf.norm(HandModel.cross_product(var, ph) + HandModel.epsilon)

    @staticmethod
    def point_to_plane_loss(plane, offset, points):
        losses = [HandModel.epsilon + tf.norm(tf.tensordot(plane, point, axes=[[0], [0]]) - offset) for point in points]
        return tf.reduce_sum(losses)

    @staticmethod
    def inner_cosine(v1, v2, v3):
        diff1 = HandModel.normalized_diff(v3, v2)
        diff2 = HandModel.normalized_diff(v1, v2)
        return HandModel.unnormalized_cosine(diff1, diff2)

    @staticmethod
    def normalized_diff(p1, p2):
        diff = tf.subtract(p1, p2)
        return diff / (HandModel.epsilon + tf.norm(diff))

    @staticmethod
    def unnormalized_cosine(p1, p2):
        return tf.tensordot(p1, p2, axes=[[0], [0]])

    @staticmethod
    def plane_versor(v1, v2, v3):
        return HandModel.cross_product(HandModel.normalized_diff(v2, v1),
                                       HandModel.normalized_diff(v3, v1))

    @staticmethod
    def cross_product(v1, v2):
        x1, y1, z1 = tf.split(v1, 3)
        x2, y2, z2 = tf.split(v2, 3)
        x3 = y1 * z2 - y2 * z1
        y3 = x2 * z1 - z2 * x1
        z3 = x1 * y2 - x2 * y1
        return tf.stack((x3, y3, z3))


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
    hand_manager = HandModel()

    # here we define the hand model setup and running procedure
    # NB: not working at all.
    def loop():
        label_data = helper_hand_lab.copy()
        resolution = helper_hand_img.shape[0:2]
        set_current_calib(calibration(intr=synth_intrinsic(resolution, (50, 50))))

        formatted_data = hand_format([ImagePoint((x * resolution[0], y * resolution[1]))
                                      for (x, y, f) in label_data])
        # compute the rotation matrix
        rotation = tr.get_rotation_matrix(axis=np.array([0, 1, 0]), angle=np.pi / 180)
        # build the optimizer
        hand_manager.setup(init_dict=hand_format([point.to_camera_model() for point in raw(formatted_data)]))

        def manager_setup_and_run():
            hand_manager.start()

        threading.Thread(target=manager_setup_and_run).start()
        current_rotation = tr.get_rotation_matrix(axis=np.array([0, 1, 0]), angle=0)
        hand_manager.input_image_data(formatted_data)
        while True:
            hand_model = raw(hand_manager.out_model)
            # rotate the 3D dataset
            center = np.average(hand_model, axis=0)
            hand_model = [np.matmul(current_rotation, elem - center) + center for elem in hand_model]
            # project it into image space
            flat_2d = [ModelPoint(elem)
                           .to_image_space()
                           .as_row()
                       for elem in hand_model]
            # normalize it before feeding to the model drawer
            flat_2d_norm = [(x / resolution[0], y / resolution[1]) for (x, y) in flat_2d]
            # feed to model drawer
            md.set_joints(hand_format(flat_2d_norm))
            current_rotation = np.matmul(current_rotation, rotation)
            time.sleep(0.04)


    threading.Thread(target=loop).start()
    root.mainloop()
