import tkinter as tk
import data_manager.path_manager as path_manager
import gui.pinpointer_canvas as pc
import numpy as np

if __name__ == "__main__":
    # Path Manager to load hand images
    pm = path_manager.PathManager()
    # The root of the GUI
    root = tk.Tk()

    # This frame will include the canvas to pinpoint hand joints
    canvas_frame = tk.Frame(root)
    canvas_frame.pack()

    # Setup the pinner
    pinner = pc.PinpointerCanvas(canvas_frame)
    # A sample image, for now...
    img = np.random.uniform(low=0.0, high=1.0, size=(400, 500, 3))

    # Load the image into the pinner canvas, ready for pinpointing
    pinner.set_bitmap(img)
    # Associate custom tags to left or right click records
    pinner.pin_token[pinner.left] = "VISIBLE"
    pinner.pin_token[pinner.right] = "OCCLUDED"
    # Associate a custom action on left click... for now print known records
    pinner.on_click[pinner.left] = lambda event: print(event.widget.click_sequence)
    pinner.pack()

    # This enables resizing, but it's quite buggy. Do it at your own risk.
    # root.bind("<Configure>", lambda event: pinner.resize(event.width-2, event.height-2))

    root.mainloop()
