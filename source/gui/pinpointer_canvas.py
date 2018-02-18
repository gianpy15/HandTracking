import tkinter as tk
from PIL import Image, ImageTk
import numpy as np


class PinpointerCanvas(tk.Canvas):
    """
    This class supports a canvas displaying bitmap RGB images (from numpy arrays)
    that interactively stores and shows clicks on the image.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the PhotoImage currently displayed
        self.current_image = None
        # the Image currently displayed. Useful for online image manipulation.
        self.__img = None
        # a sequence of records of clicks on the image.
        # format: (x, y, token) where x, y belong to [0.0, 1.0] and are relative coordinates
        # and token is a custom tag to distinguish different clicks
        self.click_sequence = []

        # binding mouse events to event handlers...
        self.bind("<Button-1>", self.__left_clicked)
        self.bind("<Button-2>", self.__right_clicked)

        # class keys to reference elements in the class dictionaries (see next attributes)
        self.left = "LEFT"
        self.right = "RIGHT"

        # radius of displayed pins as circles (in pixels)
        self.pin_radius = 5
        # the color of displayed pins
        self.pin_color = {self.left: "red",
                          self.right: "green"}
        # the tokens written in records to distinguish left and right click records
        self.pin_token = {self.left: "LEFT",
                          self.right: "RIGHT"}
        # additional actions to perform when a click is taken
        self.on_click = {}

        # tags and ids to drawn figures on the canvas
        self.__image_tag = "img"
        self.__pins_tag = "pin"
        self.__pins_dict = {}

    def reset(self):
        """
        reset the status of the canvas to default with no image
        """
        self.current_image = None
        self.__img = None
        self.__pins_dict = {}
        self.click_sequence = []
        self.delete("all")

    def set_bitmap(self, bmp):
        """
        Reset the canvas and display a new image
        :param bmp: the numpy array describing the image as a bitmap
        """
        self.reset()
        if bmp.dtype in [np.float16, np.float32, np.float64, np.float128]:
            in_bmp = np.array(bmp*255, dtype=np.int8)
        elif bmp.dtype.itemsize != 1:
            in_bmp = np.array(bmp, dtype=np.int8)
        else:
            in_bmp = bmp
        self.__img = Image.fromarray(in_bmp, mode="RGB")
        self.__update_img()

    def __left_clicked(self, event):
        self.__handle_button_event(event=event, buttonkey=self.left)

    def __right_clicked(self, event):
        self.__handle_button_event(event=event, buttonkey=self.right)

    def __handle_button_event(self, event, buttonkey):
        """
        General mouse event handling procedure
        :param event: the mouse event
        :param buttonkey: the key to be used for dictionary-like attributes
        """
        click_token = (event.x / self.winfo_width(),
                       event.y / self.winfo_height(),
                       self.pin_token[buttonkey])
        self.click_sequence.append(click_token)
        pin_id = self.create_oval(event.x - self.pin_radius,
                                  event.y - self.pin_radius,
                                  event.x + self.pin_radius,
                                  event.y + self.pin_radius,
                                  fill=self.pin_color[buttonkey],
                                  tags=self.__pins_tag)
        self.__pins_dict[pin_id] = click_token
        if buttonkey in self.on_click.keys():
            self.on_click[buttonkey](event)

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
            for pin in self.__pins_dict.keys():
                xc = self.__pins_dict[pin][0]*newwidth
                yc = self.__pins_dict[pin][1]*newheight
                self.coords(pin,
                            xc - self.pin_radius,
                            yc - self.pin_radius,
                            xc + self.pin_radius,
                            yc + self.pin_radius)
