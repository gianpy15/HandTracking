from tkinter import *

import os

from gui.model_drawer import *
from PIL import ImageTk, Image
from random import randint
import time
from hand_data_management.naming import framebase
from hand_data_management.video_loader import linear_fill


class PlayerThread:
    """
    Thread used to print video frames and corresponding labels
    """

    def __init__(self, frames, canvas, status, indexes, discard, modeldrawer=None, labels=None, fps=30):
        self.labels = np.array(labels)
        self.canvas = canvas
        self.current_fps = fps
        self.speed_mult = 1.0
        self.model_drawer = modeldrawer
        self.model_drawer.set_reactions(onclick=lambda e: self.onclick(e),
                                        onmove=lambda e: self.onmove(e),
                                        onrelease=lambda e: self.onrelease(e),
                                        onrightclick=lambda e: self.onrightclick(e))
        self.label_target_no = 0
        self.label_target_original = None
        self.label_target_initial_click = None
        self.pic_height = np.shape(frames)[1]
        self.pic_width = np.shape(frames)[2]
        self.play_flag = False
        self.frame_status_msg = status
        self.indexes = np.array([True if idx == 1 else False for idx in indexes]) if indexes is not None else None
        self.discard = discard
        # Build the frame buffer at once
        if frames[0].dtype in [np.float16, np.float32, np.float64]:
            framebuff = np.array(frames * 255, dtype=np.int8)
        elif frames[0].dtype.itemsize != 1:
            framebuff = np.array(frames, dtype=np.int8)
        else:
            framebuff = frames

        # keep track of image and photoimage, otherwise they get garbage-collected
        self.imgs = [Image.fromarray(buffer, mode="RGB") for buffer in framebuff]
        self.photoimgs = [ImageTk.PhotoImage(image=img) for img in self.imgs]

        # frame counter
        self.current_frame = 0
        self.frameslider = None

        # persistent image canvas ID to be able to update it
        self.imageid = self.make_canvas_image()
        # if labels are provided, then draw them
        if self.labels is not None and self.model_drawer is not None:
            self.model_drawer.set_joints(self.labels[self.current_frame])

        self.deleted = np.array([False for _ in range(len(framebuff))])
        self.edited = np.array([False for _ in range(len(framebuff))])

        self.discard.set("Discarded" if self.deleted[self.current_frame] else "")
        if self.indexes is not None:
            self.frame_status_msg.set(self.update_frame_status(self.indexes[self.current_frame]))

    def make_photoimage(self, buffer):
        """
        Produce a new photoimage from the given buffer.
        :param buffer: the array buffer describing the image
        """
        self.current_img = Image.fromarray(buffer, mode="RGB")
        self.current_photoimg = ImageTk.PhotoImage(image=self.current_img)
        return self.current_photoimg

    def make_canvas_image(self):
        """
        Create the image object into the canvas, call just once on setup.
        The image is initialized with the current frame.
        :return: the ID of the created canvas
        """
        return self.canvas.create_image(0, 0, anchor=NW,
                                        image=self.photoimgs[self.current_frame])

    def update_frame(self):
        """
        Update the photoimage reference of the canvas image object,
        if any label has been given, update them as well
        """
        self.canvas.itemconfig(self.imageid, image=self.photoimgs[self.current_frame])
        if self.labels is not None and self.model_drawer is not None:
            self.model_drawer.set_joints(self.labels[self.current_frame])

        if self.indexes is not None:
            interpolation_msg = self.update_frame_status(self.indexes[self.current_frame])
        else:
            interpolation_msg = 'No labels available'
        frameno_msg = "frame %d" % self.current_frame
        new_msg = "%s | %s" % (interpolation_msg, frameno_msg)
        if self.frame_status_msg.get() != new_msg:
            self.frame_status_msg.set(new_msg)

        self.discard.set("Discarded" if self.deleted[self.current_frame]
                         else "Edited" if self.edited[self.current_frame]
                         else "")
        if self.frameslider is not None:
            self.frameslider.set(self.current_frame)

    def nextframe(self, root):
        if self.speed_mult == 0:
            root.after(int(1000 // self.current_fps), self.nextframe, root)
            return
        start = time.time()
        if self.play_flag:
            # update the frame counter
            self.current_frame += 1 if self.speed_mult > 0 else -1
            self.current_frame %= len(self.photoimgs)
            # display the current photoimage
            self.update_frame()
        tot = (time.time()-start) * 1000
        # make the tkinter main loop to call after the needed time
        root.after(int(-tot + 1000 // (self.current_fps * abs(self.speed_mult))), self.nextframe, root)

    def play(self):
        self.play_flag = True

    def pause(self):
        self.play_flag = False

    def set_fps(self, newfps):
        self.current_fps = int(newfps)

    def set_speed_mult(self, mult):
        self.speed_mult = float(mult)

    def update_frame_status(self, value):
        if value:
            return "Labeled"
        return "Interpolated"

    def print_changes(self, vidname):
        fname = vidname + "-" + str(randint(0, 999)) + ".txt"
        filepath = os.path.join(framebase, fname)
        file = open(filepath, "w+")
        for i in range(len(self.deleted)):
            if self.deleted[i] == 1:
                file.write("D%d\n" % i)
            elif self.edited[i]:
                file.write("E%d;%s\n" % (i, PlayerThread.encode_labels(self.labels[i])))

    def set_changes(self):
        if not self.deleted[self.current_frame]:
            self.deleted[self.current_frame] = True
            self.discard.set("Discarded")

        elif self.deleted[self.current_frame]:
            self.deleted[self.current_frame] = False
            self.discard.set("")

    def change_visibility(self, index):
        self.labels[self.current_frame][index][2] = int(1 - self.labels[self.current_frame][index][2])
        self.edited[self.current_frame] = True
        self.update_frame()

    def onrightclick(self, event):
        if self.play_flag:
            return
        relcoords = np.array((event.x / self.canvas.winfo_width(), event.y / self.canvas.winfo_height()))
        selected = np.argmin(np.linalg.norm(self.labels[self.current_frame][:, 0:2] - relcoords, axis=1))
        self.change_visibility(selected)

    def onclick(self, event):
        if self.play_flag:
            return
        relcoords = np.array((event.x / self.canvas.winfo_width(), event.y / self.canvas.winfo_height()))
        selected = np.argmin(np.linalg.norm(self.labels[self.current_frame][:, 0:2]-relcoords, axis=1))
        self.label_target_no = selected
        self.label_target_initial_click = relcoords
        self.label_target_original = self.labels[self.current_frame][selected].copy()

    def onmove(self, event):
        relcoords = np.array((event.x / self.canvas.winfo_width(), event.y / self.canvas.winfo_height()))
        delta = np.append(relcoords - self.label_target_initial_click, 0)
        self.labels[self.current_frame][self.label_target_no] = self.label_target_original + delta
        self.model_drawer.set_joints(self.labels[self.current_frame])

    def onrelease(self, event):
        relcoords = np.array((event.x / self.canvas.winfo_width(), event.y / self.canvas.winfo_height()))
        delta = np.append(relcoords - self.label_target_initial_click, 0)
        self.labels[self.current_frame][self.label_target_no] = self.label_target_original + delta
        self.model_drawer.set_joints(self.labels[self.current_frame])
        self.edited[self.current_frame] = True
        self.label_target_initial_click = None
        self.label_target_original = None
        self.update_frame()

    def keepthis(self):
        self.edited[self.current_frame] = not self.edited[self.current_frame]
        self.update_frame()

    def keepall(self):
        for i in range(len(self.labels)):
            self.edited[i] = True
            self.deleted[i] = False
        self.update_frame()

    def set_current_frame(self, frameno):
        self.current_frame = frameno % len(self.imgs)
        self.update_frame()

    def next_fixed_frame(self, jumps=1):
        if self.indexes is None:
            return
        idx = (self.current_frame + jumps) % len(self.imgs)
        while idx != self.current_frame and (self.deleted[idx] or not (self.indexes[idx] or self.edited[idx])):
            idx = (idx + jumps) % len(self.imgs)
        self.set_current_frame(idx)

    def reinterpolate(self):
        if self.indexes is None:
            return
        taken = np.logical_and(np.logical_or(self.indexes, self.edited), np.logical_not(self.deleted))
        self.labels[np.logical_not(taken)] = None
        self.labels = linear_fill(self.labels)
        self.update_frame()

    @staticmethod
    def encode_labels(labels):
        ret = ""
        for joint in labels:
            ret += "%f;%f;%d;" % (joint[0], joint[1], joint[2])
        return ret[:-1]
