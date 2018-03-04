import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import time


class PinpointerCanvas(tk.Canvas):
    """
    This class supports a canvas displaying bitmap RGB images (from numpy arrays)
    that interactively stores and shows clicks on the image.
    """

    BB_TIME_THRESHOLD = 0.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the PhotoImage currently displayed
        self.current_image = None
        # the Image currently displayed. Useful for online image manipulation.
        self.__img = None
        self.__original_img = None
        self.__current_bb = np.array([[.0, 1.], [.0, 1.]])
        # a sequence of records of clicks on the image.
        # format: (x, y, token) where x, y belong to [0.0, 1.0] and are relative coordinates
        # and token is a custom tag to distinguish different clicks
        self.click_sequence = []

        self.__enter_bb_coords = None
        self.__enter_bb_time = time.time()
        self.__bb_selection_state = False
        self.__bb_rectangle = None
        # binding mouse events to event handlers...
        self.bind("<ButtonRelease-1>", self.__left_clicked)
        self.bind("<Button-1>", self.__enter_bb_selection_state)
        self.bind("<B1-Motion>", self.__update_bb_rect)
        self.bind("<Enter>", lambda e: self.focus_set())
        self.bind("<Escape>", lambda e: self.__reset_bb())
        # double bind for OS compatibility
        self.bind("<ButtonRelease-2>", self.__right_clicked)
        self.bind("<ButtonRelease-3>", self.__right_clicked)

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
        if bmp.dtype in [np.float16, np.float32, np.float64]:
            in_bmp = np.array(bmp*255, dtype=np.int8)
        elif bmp.dtype.itemsize != 1:
            in_bmp = np.array(bmp, dtype=np.int8)
        else:
            in_bmp = bmp
        self.__original_img = Image.fromarray(in_bmp, mode="RGB")
        self.__update_img()

    def __deactivate_bb_selection_state(self):
        self.__bb_selection_state = False
        self.delete(self.__bb_rectangle)
        self.__bb_rectangle = None

    def __reset_bb(self):
        self.__current_bb = np.array([[.0, 1.], [.0, 1.]])
        self.__deactivate_bb_selection_state()
        self.__update_img()

    def __update_bb_rect(self, event):
        if not self.__bb_selection_state or time.time() - self.__enter_bb_time < PinpointerCanvas.BB_TIME_THRESHOLD:
            return
        rectx1 = self.__enter_bb_coords[0] * self.winfo_width()
        recty1 = self.__enter_bb_coords[1] * self.winfo_height()
        if self.__bb_rectangle is None:
            self.__bb_rectangle = self.create_rectangle(rectx1, recty1,
                                                        event.x, event.y,
                                                        outline="red",
                                                        dash=(5,))
        else:
            self.coords(self.__bb_rectangle, rectx1, recty1, event.x, event.y)

    def __left_clicked(self, event):
        if not self.__bb_selection_state or time.time()-self.__enter_bb_time < PinpointerCanvas.BB_TIME_THRESHOLD:
            self.__handle_button_event(event=event, buttonkey=self.left)
        else:
            end_bb_coords = event.x / self.winfo_width(), event.y / self.winfo_height()
            self.__enter_bb_coords = PinpointerCanvas.to_global_relative_coords(self.__current_bb,
                                                                                self.__enter_bb_coords)
            end_bb_coords = PinpointerCanvas.to_global_relative_coords(self.__current_bb, end_bb_coords)
            xcoords = self.__enter_bb_coords[0], end_bb_coords[0]
            ycoords = self.__enter_bb_coords[1], end_bb_coords[1]
            self.__current_bb = np.array([[min(xcoords), max(xcoords)], [min(ycoords), max(ycoords)]])
            self.__deactivate_bb_selection_state()
            self.__update_img()

    def __enter_bb_selection_state(self, event):
        self.__enter_bb_coords = event.x / self.winfo_width(), event.y / self.winfo_height()
        self.__enter_bb_time = time.time()
        self.__bb_selection_state = True

    def __right_clicked(self, event):
        self.__handle_button_event(event=event, buttonkey=self.right)

    def __handle_button_event(self, event, buttonkey):
        """
        General mouse event handling procedure
        :param event: the mouse event
        :param buttonkey: the key to be used for dictionary-like attributes
        """
        rel_coords = event.x / self.winfo_width(), event.y / self.winfo_height()
        glob_coords = PinpointerCanvas.to_global_relative_coords(self.__current_bb, rel_coords)
        click_token = (glob_coords[0],
                       glob_coords[1],
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
        Synchronize the self.current_image according to the self.__img and current bounding box
        Must call after manipulating self.__original_img or self.__current_bb to update the view
        """
        self.delete(self.__image_tag)
        cur_bb = np.empty(shape=(4,), dtype=np.uint16)
        cur_bb[0] = self.__current_bb[0][0] * self.__original_img.width
        cur_bb[1] = self.__current_bb[1][0] * self.__original_img.height
        cur_bb[2] = self.__current_bb[0][1] * self.__original_img.width
        cur_bb[3] = self.__current_bb[1][1] * self.__original_img.height
        w_mult = self.__original_img.width / (cur_bb[2] - cur_bb[0])
        h_mult = self.__original_img.height / (cur_bb[3] - cur_bb[1])
        mult = min(w_mult, h_mult)
        size = int((cur_bb[2]-cur_bb[0]) * mult), int((cur_bb[3] - cur_bb[1]) * mult)
        self.__img = self.__original_img.resize(size=size, box=cur_bb, resample=Image.BILINEAR)
        self.current_image = ImageTk.PhotoImage(image=self.__img)
        self.configure(width=self.current_image.width(), height=self.current_image.height())
        self.create_image(0, 0, image=self.current_image, anchor=tk.NW, tags=self.__image_tag)
        self.__update_pin_relative_position()
        self.tag_lower(self.__image_tag)

    def __update_pin_relative_position(self):
        for pin in self.__pins_dict.keys():
            rel_pnt = PinpointerCanvas.to_bonuded_relative_coords(self.__current_bb, self.__pins_dict[pin][0:2])
            rel_pnt[0] *= self.__img.width
            rel_pnt[1] *= self.__img.height
            self.coords(pin,
                        rel_pnt[0] - self.pin_radius,
                        rel_pnt[1] - self.pin_radius,
                        rel_pnt[0] + self.pin_radius,
                        rel_pnt[1] + self.pin_radius)

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

    @staticmethod
    def to_global_relative_coords(current_bb, pnt):
        return np.array([pnt[i] * (current_bb[i][1] - current_bb[i][0]) + current_bb[i][0] for i in range(2)])

    @staticmethod
    def to_bonuded_relative_coords(current_bb, globpnt):
        return np.array([(globpnt[i] - current_bb[i][0]) / (current_bb[i][1] - current_bb[i][0]) for i in range(2)])
