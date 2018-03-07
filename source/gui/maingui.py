import tkinter as tk
import data_manager.path_manager as path_manager
import image_loader.image_loader as il
import gui.pinpointer_canvas as pc
import gui.hand_helper_canvas as hh
import scipy.io as scio
import image_loader.hand_io as hio
import skimage.transform as skt
import numpy as np

helper_ref = None
helptext = "Click on the image on the left to set the position of the joints of the hand.\nLeft click for visible " \
           "joints, right click for occluded joints."


def setup_pinner(pinner, img, depth):
    data = img

    def joints_counter(event):
        if helper_ref is not None:
            helper_ref.next()
        if len(event.widget.click_sequence) == 21:
            save_labels(event.widget.click_sequence)

    def save_labels(labels):
        hio.store(pm.resources_path("calibration_apply_test.mat"), data=data, labels=labels, depth=depth)

    # Load the image into the pinner canvas, ready for pinpointing
    pinner.set_bitmap(img)
    # Associate custom tags to left or right click records
    pinner.pin_token[pinner.left] = 0
    pinner.pin_token[pinner.right] = 1
    # Associate a custom action on left click... for now print known records
    pinner.on_click[pinner.left] = joints_counter
    pinner.on_click[pinner.right] = joints_counter
    pinner.pack()


def setup_helper(helper, data):
    helper.set_bitmap_and_labels(image=data['data'], records=data['labels'])
    helper.pack(side=tk.BOTTOM)


if __name__ == "__main__":
    # Path Manager to load hand images
    pm = path_manager.PathManager()
    # The root of the GUI
    root = tk.Tk()

    # This frame will include the canvas to pinpoint hand joints
    canvas_frame = tk.Frame(root)
    canvas_frame.pack(side=tk.LEFT)

    # This frame will include all side commands
    side_frame = tk.Frame(root)
    side_frame.pack(side=tk.RIGHT, fill=tk.BOTH)

    # A sample image, for now...
    # img = il.load(pm.resources_path("gui/Flower-bud-003.jpg"), force_format=[None, None, 3])[0]
    img, depth = hio.load(pm.resources_path("gui/hand.mat"), format=(hio.RGB_DATA, hio.DEPTH_DATA))

    # off = (img.shape[0]-img.shape[1])//2
    # img = np.pad(img, pad_width=((0, 0), (off, off), (0, 0)), mode='constant')
    # img = skt.resize(img, output_shape=(700, 700))

    # Setup the pinner
    pinner = pc.PinpointerCanvas(canvas_frame)
    setup_pinner(pinner, img, depth)

    # load the helper data into a dictionary
    # helper_hand['data'] is the image
    # helper_hand['labels'] are the points
    helper_hand = scio.loadmat(pm.resources_path("gui/sample_hand.mat"))

    # Setup the helper
    helper = hh.HelperCanvas(side_frame)
    setup_helper(helper, helper_hand)
    helper_ref = helper
    descriptor = tk.Message(side_frame,
                            text=helptext,
                            justify=tk.LEFT,
                            font=('Arial', 10, 'bold'))
    descriptor.pack(side=tk.TOP)

    # This enables resizing, but it's quite buggy. Do it at your own risk.
    # root.bind("<Configure>", lambda event: pinner.resize(event.width-2, event.height-2))

    root.mainloop()
