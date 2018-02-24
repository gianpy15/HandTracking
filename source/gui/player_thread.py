from tkinter import *
from gui.model_drawer import *
from PIL import ImageTk, Image


class PlayerThread:
    """
    Thread used to print video frames and corresponding labels
    """

    def __init__(self, frames, canvas, modeldrawer=None, labels=None, fps=60):
        self.labels = labels
        self.canvas = canvas
        self.current_fps = fps
        self.model_drawer = modeldrawer
        self.pic_height = 168
        self.pic_width = 298
        self.play_flag = False
        # Build the frame buffer at once
        self.framebuff = []
        if frames[0].dtype in [np.float16, np.float32, np.float64, np.float128]:
            self.framebuff = np.array(frames * 255, dtype=np.int8)
        elif frames[0].dtype.itemsize != 1:
            self.framebuff = np.array(frames, dtype=np.int8)
        else:
            self.framebuff = frames

        # keep track of image and photoimage, otherwise they get garbage-collected
        self.current_photoimg = None
        self.current_img = None

        # frame counter
        self.current_frame = 0

        # persistent image canvas ID to be able to update it
        self.imageid = self.make_canvas_image()
        # if labels are provided, then draw them
        if self.labels is not None and self.model_drawer is not None:
            self.model_drawer.set_joints(self.labels[self.current_frame])

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
                                        image=self.make_photoimage(self.framebuff[self.current_frame]))

    def update_frame(self):
        """
        Update the photoimage reference of the canvas image object,
        if any label has been given, update them as well
        """
        self.canvas.itemconfig(self.imageid, image=self.current_photoimg)
        if self.labels is not None and self.model_drawer is not None:
            self.model_drawer.set_joints(self.labels[self.current_frame])

    def nextframe(self, root):
        if self.play_flag:
            # update the frame counter
            self.current_frame += 1 if self.current_fps > 0 else -1
            self.current_frame %= len(self.framebuff)
            # update self.current_photoimage
            self.make_photoimage(self.framebuff[self.current_frame])
            # display the current photoimage
            self.update_frame()
        # make the tkinter main loop to call after the needed time
        root.after(1000 // abs(self.current_fps), self.nextframe, root)

    def play(self):
        self.play_flag = True

    def pause(self):
        self.play_flag = False

