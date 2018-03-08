from PIL import Image, ImageTk
import tkinter as tk


class FrameDisplayer:
    def __init__(self, canvas, mode):
        self.canvas = canvas
        self.mode = mode
        self.current_img = None
        self.current_photoimg = None
        self.canvas_img = None

    def make_photoimage(self, buffer):
        """
        Produce a new photoimage from the given buffer.
        :param buffer: the array buffer describing the image
        """
        # self.current_img = Image.frombuffer(mode=self.mode, size=(640, 480), data=buffer)
        self.current_img = Image.fromarray(buffer, self.mode)
        self.current_photoimg = ImageTk.PhotoImage(image=self.current_img)
        return self.current_photoimg

    def make_canvas_image(self):
        """
        Create the image object into the canvas, call just once on setup.
        The image is initialized with the current frame.
        :return: the ID of the created canvas
        """
        return self.canvas.create_image(0, 0, anchor=tk.NW,
                                        image=self.current_photoimg)

    def update_frame(self, buffer):
        self.make_photoimage(buffer)
        if self.canvas_img is None:
            self.make_canvas_image()
        else:
            self.canvas.itemconfig(self.canvas_img, image=self.current_photoimg)

