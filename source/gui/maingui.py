import tkinter as tk
import data_manager.path_manager as path_manager
import image_loader.image_loader as il
import gui.pinpointer_canvas as pc
import gui.hand_helper_canvas as hh
import scipy.io as scio


helper_ref = None


def setup_pinner(pinner, img):
    data = img

    def joints_counter(event):
        if helper_ref is not None:
            helper_ref.next()
        if len(event.widget.click_sequence) == 21:
            save_labels(event.widget.click_sequence)

    def save_labels(labels):
        il.save_mat(pm.resources_path("gui/hand.mat"), data=data, labels=labels)

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
    side_frame.pack(side=tk.RIGHT)

    # A sample image, for now...
    # img = np.random.uniform(low=0.0, high=1.0, size=(400, 500, 3))
    img = il.load(pm.resources_path("gui/hand.jpg"), force_format=[None, None, 3])[0]

    # Setup the pinner
    pinner = pc.PinpointerCanvas(canvas_frame)
    setup_pinner(pinner, img)

    # load the helper data into a dictionary
    # helper_hand['data'] is the image
    # helper_hand['labels'] are the points
    helper_hand = scio.loadmat(pm.resources_path("gui/sample_hand.mat"))

    # Setup the helper
    helper = hh.HelperCanvas(side_frame)
    setup_helper(helper, helper_hand)
    helper_ref = helper


    # This enables resizing, but it's quite buggy. Do it at your own risk.
    # root.bind("<Configure>", lambda event: pinner.resize(event.width-2, event.height-2))

    root.mainloop()
