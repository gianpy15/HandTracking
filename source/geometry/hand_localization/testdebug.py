from geometry.formatting import *
from numpy.linalg import norm
from geometry.hand_localization.hand_localization_num import compute_hand_world_joints


def end_check(model1, model2):
    """
    DEBUG UTILITY
    Check whether the two given models have any length differences
    """

    def segment_len(model, finger, idx):
        return norm(model[finger][idx] - model[finger][idx + 1])

    def inconsistency(finger, idx):
        return np.abs(segment_len(model1, finger, idx) - segment_len(model2, finger, idx))

    for finger in FINGERS:
        for idx in range(3):
            if inconsistency(finger, idx) > 1e-4:
                print("Inconsistency in %s %d by %f" % (finger, idx, inconsistency(finger, idx)))


def drawpnts(ptd, cal, canvas, fill=None):
    ptd = [ModelPoint(pt).to_image_space(cal).as_row() for pt in ptd]
    for pt in ptd:
        canvas.create_oval(pt[0] - 5,
                           pt[1] - 5,
                           pt[0] + 5,
                           pt[1] + 5,
                           fill=fill,
                           tag="debug")


# testing with some fancy animations!
if __name__ == '__main__':
    import tkinter as tk
    import image_loader.hand_io as hio
    import gui.pinpointer_canvas as ppc
    import time
    import threading
    import geometry.transforms as tr
    from geometry.calibration import *
    from gui.model_drawer import ModelDrawer
    from geometry.default_model_loading import build_default_hand_model
    from timeit import timeit
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np

    REPETITIONS = 100
    EXTRATHREADS = 0
    PRESLEEP = 1
    INFILE = "gui/hand.mat"

    root = tk.Tk()
    frame = tk.Frame(root)
    frame.pack()
    subj_img, subj_lab = hio.load(INFILE)
    # subj_lab = np.array([(1-x, y, f) for (x, y, f) in subj_lab])
    test_subject = ppc.PinpointerCanvas(frame)
    test_subject.set_bitmap(subj_img)
    test_subject.pack()
    global_canvas = test_subject
    md = ModelDrawer()
    md.set_joints(hand_format(subj_lab))
    md.set_target_area(test_subject)

    if EXTRATHREADS > 0:
        pool = ThreadPoolExecutor(EXTRATHREADS)
    else:
        pool = None

    # here we define the hand model setup and running procedure
    # NB: not working at all.
    def loop():
        label_data = subj_lab.copy()
        resolution = subj_img.shape[0:2]
        fov = 40
        cal = calibration(intr=synth_intrinsic(resolution, (fov, fov * subj_img.shape[1] / subj_img.shape[0])))
        global global_calibration
        global_calibration = cal

        formatted_data = hand_format([ImagePoint((x * resolution[1], y * resolution[0]))
                                      for (x, y, f) in label_data])
        # compute the rotation matrix
        rotation = tr.get_rotation_matrix(axis=np.array([0, 1, 0]), angle=np.pi / 180)

        def total():
            global hand_model
            hand_model = raw(compute_hand_world_joints(build_default_hand_model(),
                                                       formatted_data,
                                                       cal=cal,
                                                       executor=pool))

            # make sure that the GUI-related load is expired before measuring performance

        time.sleep(PRESLEEP)
        print("Model computation %d times took %f seconds." % (REPETITIONS, timeit(total, number=REPETITIONS)))

        current_rotation = tr.get_rotation_matrix(axis=np.array([0, 1, 0]), angle=0)
        time.sleep(1)
        while True:
            # rotate the 3D dataset
            center = np.average(hand_model, axis=0)
            rotated_model = [np.matmul(current_rotation, elem - center) + center for elem in hand_model]
            # project it into image space
            flat_2d = [ModelPoint(elem).to_image_space(calibration=cal, makedepth=True)
                       for elem in rotated_model]
            # normalize it before feeding to the model drawer
            # feed to model drawer
            md.set_joints(flat_2d, resolution=resolution)
            current_rotation = np.matmul(current_rotation, rotation)
            time.sleep(0.02)


    threading.Thread(target=loop).start()
    root.mainloop()
