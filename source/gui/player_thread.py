from multiprocessing import *
from time import sleep
import numpy as np
from geometry.formatting import hand_format
from tkinter import *
from gui.model_drawer import *
from PIL import ImageTk, Image


class PlayerThread(Process):
    """
    Thread used to print video frames and corresponding labels
    """

    def __init__(self,frames, labels, canvas, md, flag):
        Process.__init__(self)

        self.delay = .01
        self.frames = frames
        self.labels = labels
        self.canvas = canvas
        self.model_drawer = md
        self.pic_height = 168
        self.pic_width = 298
        self.play_flag = flag

    def run(self):
        if np.array(self.frames).dtype in [np.float16, np.float32, np.float64, np.float128]:
            frames_list = np.array(self.frames * 255, dtype=np.int8)
        elif np.array(self.frames).dtype.itemsize != 1:
            frames_list = np.array(self.frames, dtype=np.int8)
        else:
            frames_list = self.frames

        self.play_flag.set()
        for i in range(len(frames_list)):
            self.play_flag.wait()
            image = Image.fromarray(frames_list[i], mode="RGB")
            photo_image = ImageTk.PhotoImage(image)
            img = self.canvas.create_image(self.pic_width, self.pic_height, image=photo_image)
            self.model_drawer.set_joints(hand_format(self.labels[i]))
            print(i)
            n = self.delay
            #sleep(n)
            #while n > 0:
            #    n = n - 1
            self.canvas.update()

    def set_delay(self, value):
        self.delay = value