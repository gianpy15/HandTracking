from geometry.formatting import *
from geometry.calibration import *
import re
import numpy as np


class ModelDrawer:
    """
    This class manages the objects related to drawing a hand model into a canvas.
    It can manage one canvas at a time and takes care of canvas object instantiation,
    update and deletion when needed.
    """
    key_rex = re.compile('[^-]+-[^-]+')

    def __init__(self):
        self.canvas = None
        self.posx = None
        self.posy = None
        self.width = None
        self.height = None
        self.joints = {}
        self.drawn = {}
        self.depthwise_ordering = []
        self.need_depth_update = False

        self.__all_tag = 'allmd'
        self.__dots_tag = 'dotmd'
        self.__lines_tag = 'linemd'

        self.dot_colors = {
            WRIST: "black",
            THUMB: "green",
            INDEX: "yellow",
            MIDDLE: "red",
            RING: "purple",
            BABY: "blue"
        }
        self.dot_radius = 5

    def set_target_area(self, canvas, position=(0, 0), size=None):
        """
        Set the canvas where the drawing will be performed and the region of the canvas
        where the points should be drawn
        :param canvas: the target canvas of the drawing
        :param position: the offset to give to points with respect to the canvas. Defaults to zero.
        :param size: the size we want our drawn window to have. Defaults to canvas whole dimensions.
        """
        if size is None:
            self.width = canvas.winfo_width()
            self.height = canvas.winfo_height()
            canvas.bind("<Configure>", lambda e: self.set_target_area(e.widget, position))
        else:
            self.width = size[0]
            self.height = size[1]

        self.posx = position[0]
        self.posy = position[1]

        if self.canvas != canvas:
            if self.canvas is not None:
                self.canvas.delete(self.__all_tag)
            self.canvas = canvas
            self.drawn = {}

        if self.__joints_set():
            self.draw_model()

    def set_joints(self, joints, resolution=(1.0, 1.0)):
        """
        Set or update the joints data to be drawn, and automagically draw them.
        :param joints: the hand joints in rich dictionary standard format (see geometry.formatting)
        """
        if isinstance(joints, dict):
            joints = raw(joints)
        if isinstance(joints[0], ImagePoint):
            new_depthwise_ordering = np.argsort([p.depth
                                                 if p.depth is not None
                                                 else np.inf
                                                 for p in joints])[::-1]
            if np.min(new_depthwise_ordering) != np.inf and \
                    (len(self.depthwise_ordering) < 21 or
                     np.any(self.depthwise_ordering - new_depthwise_ordering)):
                self.depthwise_ordering = new_depthwise_ordering
                self.need_depth_update = True
            self.joints = hand_format([(p.coords[0] / resolution[0], p.coords[1] / resolution[1]) for p in joints])
        else:
            self.joints = hand_format([(p[0] / resolution[0], p[1] / resolution[1]) for p in joints])
        if self.canvas is not None:
            self.draw_model()

    def draw_model(self):
        """
        Draw the model. Already called by set_joints.
        """
        if self.__drawn_anything():
            self.__update_draw()
        else:
            self.__draw_anew()
        if self.need_depth_update:
            self.__top_depthwise()
            self.need_depth_update = False

    def __draw_anew(self):
        # draw the wrist
        self.drawn[WRIST] = {}
        self.__draw_new_joint(WRIST)
        # draw the main fingers:
        for finger in FINGERS:
            self.drawn[finger] = {}
            # draw the first three joints and their outgoing section
            for joint in range(3):
                self.__draw_new_joint(finger, joint)
                self.__draw_new_section(finger, joint)
            # end up the finger drawing the last top joint
            self.__draw_new_joint(finger, 3)
            # and finally connect the wrist with that finger
            self.drawn["%s-%s" % (WRIST, finger)] = self.__draw_custom_line(
                self.joints[WRIST][0],
                self.joints[finger][0],
                self.dot_colors[WRIST]
            )
        # now make the hand palm with lines connecting the base finger joints
        for idx in range(len(FINGERS) - 1):
            key = "%s-%s" % (FINGERS[idx], FINGERS[idx + 1])
            self.drawn[key] = self.__draw_custom_line(
                self.joints[FINGERS[idx]][0],
                self.joints[FINGERS[idx + 1]][0],
                self.dot_colors[WRIST]
            )
        # and finally set all points on the top layer
        self.canvas.tag_raise(self.__dots_tag)

    def __update_draw(self):
        # update the wrist
        self.__update_joint(WRIST)
        for finger in FINGERS:
            # now update the base joints
            for joint in range(4):
                self.__update_joint(finger, joint)
            # and update the finger sections
            for joint in range(3):
                key = "%d-%d" % (joint, joint + 1)
                self.__update_line(self.drawn[finger][key],
                                   self.joints[finger][joint],
                                   self.joints[finger][joint + 1])
        # then find out all base lines
        linelist = [[key] + key.split('-') for key in self.drawn.keys() if re.match(ModelDrawer.key_rex, key)]
        # and adjust them
        for (compound_key, key1, key2) in linelist:
            self.__update_line(self.drawn[compound_key],
                               self.joints[key1][0],
                               self.joints[key2][0])

    def __draw_new_joint(self, finger, joint=0):
        drawx, drawy = self.__get_coordinates(self.joints[finger][joint])
        self.drawn[finger][joint] = self.canvas.create_oval(drawx - self.dot_radius,
                                                            drawy - self.dot_radius,
                                                            drawx + self.dot_radius,
                                                            drawy + self.dot_radius,
                                                            fill=self.dot_colors[finger],
                                                            tags=[self.__all_tag, self.__dots_tag]
                                                            )

    def __draw_new_section(self, finger, basejoint):
        self.drawn[finger]["%d-%d" % (basejoint, basejoint + 1)] = self.__draw_custom_line(
            self.joints[finger][basejoint],
            self.joints[finger][basejoint + 1],
            self.dot_colors[finger]
        )

    def __draw_custom_line(self, point1, point2, color):
        drawx1, drawy1 = self.__get_coordinates(point1)
        drawx2, drawy2 = self.__get_coordinates(point2)
        return self.canvas.create_line(drawx1,
                                       drawy1,
                                       drawx2,
                                       drawy2,
                                       fill=color,
                                       width=self.dot_radius / 2,
                                       tags=[self.__all_tag, self.__lines_tag]
                                       )

    def __update_joint(self, finger, joint=0):
        drawx, drawy = self.__get_coordinates(self.joints[finger][joint])
        self.canvas.coords(self.drawn[finger][joint],
                           drawx - self.dot_radius,
                           drawy - self.dot_radius,
                           drawx + self.dot_radius,
                           drawy + self.dot_radius
                           )

    def __update_line(self, lineid, point1, point2):
        drawx1, drawy1 = self.__get_coordinates(point1)
        drawx2, drawy2 = self.__get_coordinates(point2)
        self.canvas.coords(lineid,
                           drawx1,
                           drawy1,
                           drawx2,
                           drawy2,
                           )

    def __top_depthwise(self):
        if len(self.depthwise_ordering) < 21:
            return
        drawnkeys = self.drawn.keys()
        for idx in self.depthwise_ordering:
            finger, index = get_label_and_index(idx)
            for key in drawnkeys:
                if re.match("%s-[^-]+" % finger, key):
                    self.canvas.tag_raise(self.drawn[key])
            if finger != WRIST:
                for key in self.drawn[finger].keys():
                    if re.match("[^-]+-%d" % index, str(key)):
                        self.canvas.tag_raise(self.drawn[finger][key])
            self.canvas.tag_raise(self.drawn[finger][index])

    def __get_coordinates(self, point):
        return self.posx + self.width * point[0], self.posy + self.height * point[1]

    def __joints_set(self):
        return len(self.joints.keys()) > 0

    def __drawn_anything(self):
        return len(self.drawn.keys()) > 0


# testing with some fancy animations!
if __name__ == '__main__':
    import tkinter as tk
    import image_loader.hand_io as hio
    import gui.pinpointer_canvas as ppc
    import time
    import threading
    import geometry.transforms as tr
    from geometry.calibration import *
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


    # here we define the flat rotation with random constellation
    # points must appear correctly on the image, but their distance should be
    # different and randomized (the constellation effect)
    # and then they should rotate rigidly around their center
    def loop():
        label_data = helper_hand_lab.copy()
        resolution = helper_hand_img.shape[0:2]
        camera_calib = calibration(intr=synth_intrinsic(resolution, (50, 50)))
        # Here we arbitrarily set their depth to make the constellation effect
        flat_3d = [ImagePoint((elem[0] * resolution[0], elem[1] * resolution[1]), depth=10 + np.random.random())
                       .to_camera_model(calibration=camera_calib)
                       .as_row()
                   for elem in label_data]
        # compute the center of the constellation
        center = np.average(flat_3d, axis=0)
        # compute the rotation matrix
        rotation = tr.get_rotation_matrix(axis=1, angle=np.pi / 180)

        while True:
            # rotate the 3D dataset
            flat_3d = [np.matmul(rotation, elem - center) + center for elem in flat_3d]
            # project it into image space
            flat_2d = [ModelPoint(elem)
                           .to_image_space(calibration=camera_calib)
                           .as_row()
                       for elem in flat_3d]
            # normalize it before feeding to the model drawer
            flat_2d_norm = [(x / resolution[0], y / resolution[1]) for (x, y) in flat_2d]
            # feed to model drawer
            md.set_joints(hand_format(flat_2d_norm))
            time.sleep(0.04)


    threading.Thread(target=loop).start()
    root.mainloop()
