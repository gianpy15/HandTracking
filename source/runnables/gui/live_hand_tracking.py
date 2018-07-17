import time

import cv2
import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..")))
from library.neural_network import heatmap, preprocess_input
from skimage.transform import resize
from library.utils.visualization_utils import get_image_with_mask
import numpy as np
from library.utils.hsv import rgb2hsv, hsv2rgb


def preprocess_frame(frame):
    _frame = resize(frame, output_shape=(224, 224))
    # in case your webcam is BGR...
    _frame = np.flip(_frame, axis=2)
    rgb2hsv(_frame)
    _frame[:, :, 1] += 0.65
    _frame[:, :, 0] += 0.25
    _frame = np.clip(_frame, a_min=0, a_max=1)
    print("Saturation mean: {}".format(np.mean(_frame[:, :, 1])))
    print("Hue mean: {}".format(np.mean(_frame[:, :, 0])))
    hsv2rgb(_frame)
    _frame = np.expand_dims(_frame, axis=0)
    _frame = preprocess_input(_frame * 255)
    return _frame


if __name__ == '__main__':
    net = heatmap()

    cap = cv2.VideoCapture(0)  # Capture video from camera

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use the lower case
    out = cv2.VideoWriter('output.mp4', fourcc, 5.0, (width, height))

    recording = False
    i_time = time.time()
    count = 0

    while cap.isOpened():
        l_time = time.time()
        count += 1
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_ = preprocess_frame(frame)
            mask = net.predict(frame_)[0]
            frame = np.array(get_image_with_mask(frame, mask) / 255)

            # write the flipped frame
            if recording:
                out.write(frame)

            cv2.imshow('frame', frame)

            if (cv2.waitKey(1) & 0xFF) == ord('r'):
                print("Recording..." if not recording else "Stop Recording.")
                recording = not recording

            if (cv2.waitKey(1) & 0xFF) == ord('q'):  # Hit `q` to exit
                break
        else:
            break
        print("{} fps (mean {} fps)".format(1/(time.time()-l_time), 1/((time.time()-i_time)/count)))

    # Release everything if job is finished
    out.release()
    cap.release()
    cv2.destroyAllWindows()
