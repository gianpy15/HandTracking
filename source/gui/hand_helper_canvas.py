import tkinter as tk
from PIL import Image, ImageTk
import numpy as np


class HelperCanvas(tk.Canvas):
    """
    This class supports a canvas displaying bitmap RGB images (from numpy arrays)
    and given relative points to be displayed in order one after the other
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the PhotoImage currently displayed
        self.current_image = None
        # the Image currently displayed. Useful for online image manipulation.
        self.__img = None
        # the sequence of records to be displayed on the image.
        # format: (x, y, token) where x, y belong to [0.0, 1.0] and are relative coordinates
        # and token is a custom tag to distinguish different clicks
        self.record_sequence = []

        self.bind("<Configure>", lambda e: self.resize(e.width-2, e.height-2))

        # class keys to reference elements in the class dictionaries (see next attributes)
        self.old = "OLD"
        self.present = "PRESENT"

        # radius of displayed pins as circles (in pixels)
        self.pin_radius = 5
        # the color of displayed pins
        self.pin_color = {self.old: "blue",
                          self.present: "yellow"}

        # tags and ids to drawn figures on the canvas
        self.__image_tag = "img"
        self.__pins_tag = "pin"
        self.__displayed_list = []
        # the index of the last displayed pin
        self.__current_idx = -1

    def reset(self):
        """
        reset the status of the canvas to default with no image
        """
        self.current_image = None
        self.__img = None
        self.__displayed_list = []
        self.record_sequence = []
        self.__current_idx = -1
        self.delete("all")

    def set_bitmap_and_labels(self, image, records):
        """
        Reset the canvas and display a new image
        :param image: the numpy array describing the image as a bitmap
        :param records: the record points to be displayed in sequence.
                format: [(x, y, ...), (x, y, ...) ...]
                        x, y: float in [0.0, 1.0]
                        NB: further elements per tuple are permitted but ignored
        """
        self.reset()
        if image.dtype in [np.float16, np.float32, np.float64]:
            in_bmp = np.array(image*255, dtype=np.int8)
        elif image.dtype.itemsize != 1:
            in_bmp = np.array(image, dtype=np.int8)
        else:
            in_bmp = image
        self.__img = Image.fromarray(in_bmp, mode="RGB")
        self.__update_img()
        self.record_sequence = [elem[0:2] for elem in records]
        self.next()

    def __update_img(self):
        """
        Synchronize the self.current_image according to the self.__img.
        Must call after manipulating self.__img to update the view
        """
        self.delete(self.__image_tag)
        self.current_image = ImageTk.PhotoImage(image=self.__img)
        self.configure(width=self.current_image.width(), height=self.current_image.height())
        self.create_image(0, 0, image=self.current_image, anchor=tk.NW, tags=self.__image_tag)
        self.tag_lower(self.__image_tag)

    def resize(self, newwidth, newheight):
        """
        Perform a resize of the canvas resizing the current image and
        preserving pin relative positions
        """
        self.configure(width=newwidth, height=newheight)
        if self.__img is not None:
            self.__img = self.__img.resize(size=(newwidth, newheight), resample=Image.BILINEAR)
            self.__update_img()
            for pin in range(0, self.__current_idx+1):
                xc = self.record_sequence[pin][0]*newwidth
                yc = self.record_sequence[pin][1]*newheight
                self.coords(self.__displayed_list[pin],
                            xc - self.pin_radius,
                            yc - self.pin_radius,
                            xc + self.pin_radius,
                            yc + self.pin_radius)
        # Currently this function is used for startup setup only.
        # As it is buggy with repeated use, it deactivates itself
        self.unbind("<Configure>")

    def next(self):
        if self.__current_idx >= 0:
            self.itemconfigure(self.__displayed_list[self.__current_idx], fill=self.pin_color[self.old])

        self.__current_idx += 1
        if self.__current_idx < len(self.record_sequence):
            xrel, yrel = self.record_sequence[self.__current_idx]
            xc = xrel * self.winfo_width()
            yc = yrel * self.winfo_height()
            pin_id = self.create_oval(xc - self.pin_radius,
                                      yc - self.pin_radius,
                                      xc + self.pin_radius,
                                      yc + self.pin_radius,
                                      fill=self.pin_color[self.present],
                                      tags=self.__pins_tag)
            self.__displayed_list.append(pin_id)
