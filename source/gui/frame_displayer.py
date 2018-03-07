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
        self.current_img = Image.fromarray(buffer, mode=self.mode)
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


if __name__ == '__main__':
    import numpy as np
    import timeit
    root = tk.Tk()
    root.title("Test")

    width = 640
    height = 480

    canvas = tk.Canvas(root, width=width, height=height)
    canvas.pack()

    fd = FrameDisplayer(canvas, mode="I")

    current_col = 0

    def periodic_action():
        global current_col
        frame = current_col * np.ones(shape=(height, width), dtype=np.uint32)
        current_col = (current_col + 1) % (2 << 8)

        def totime():
            fd.update_frame(frame)

        print(1000 * timeit.timeit(totime, number=1))
        root.after(1, periodic_action)

    root.after(1000, periodic_action)
    root.mainloop()
