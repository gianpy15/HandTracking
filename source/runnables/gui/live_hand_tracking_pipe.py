import cv2
import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..", "..")))
from library.neural_network import heatmap, preprocess_input
from skimage.transform import resize
from library.utils.visualization_utils import get_image_with_mask
import numpy as np
from library.utils.hsv import rgb2hsv, hsv2rgb
from library.load_management.operational_module import OperationalModule, NoOutputException
from time import time

from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk
from skimage.draw import polygon_perimeter
from skimage.exposure import equalize_hist as equalize


def build_border(bbox, frameshape):
    np.clip(bbox, a_min=0.01, a_max=0.99, out=bbox)
    bbox[0] *= frameshape[0]
    bbox[1] *= frameshape[1]
    bbox = np.array(bbox, dtype=np.uint16)
    rows = (bbox[0, 0], bbox[0, 1], bbox[0, 1], bbox[0, 0])
    cols = (bbox[1, 0], bbox[1, 0], bbox[1, 1], bbox[1, 1])
    return polygon_perimeter(rows, cols, shape=frameshape, clip=True)


def preprocess_frame(frame):
    _frame = resize(frame, output_shape=(224, 224))
    _frame = np.flip(_frame, axis=2)
    _frame = equalize(_frame)
    _frame = np.expand_dims(_frame, axis=0)
    _frame = preprocess_input(_frame * 255)
    return _frame


def extract_position(input, output):
    output = output[0, :, :, 0]
    bw = closing(output > 0.15, disk(5))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)

    # extract the greatest area label
    areas = sorted(regionprops(label_image), key=lambda x: x.area)
    if not len(areas):
        raise NoOutputException
    maxarea = areas[-1]
    if maxarea.area < 20:
        raise NoOutputException
    bbox = np.array(maxarea.bbox).reshape(2, 2).T
    outputshape = np.array(output.shape[:2])
    bbox = np.divide(bbox, outputshape[:, None].T)
    return bbox


if __name__ == '__main__':
    net = heatmap()

    cap = cv2.VideoCapture(0)  # Capture video from camera

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use the lower case
    out = cv2.VideoWriter('output.mp4', fourcc, 5.0, (width, height))

    recording = False
    working_frequency = 5.0

    def provide_cap(time):
        ret, frame = cap.read()
        if ret:
            return [preprocess_frame(cv2.flip(frame, 1))], {}
        return None

    tracker = OperationalModule(func=net.predict, workers=3,
                                input_source=provide_cap,
                                output_adapter=extract_position,
                                working_frequency=working_frequency,
                                interp_order=0,
                                interp_samples=1)
    tracker.start()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            t = time()
            frame = cv2.flip(frame, 1)
            bbox = tracker[t-tracker.latency]
            border = build_border(bbox, frame.shape)
            # print(border)
            frame[border] = (0, 0, 255)

            cv2.putText(frame, "Net operating frequency: %.2f Hz" % tracker.frequency,
                        org=(20, height-20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 0, 255))
            cv2.putText(frame, "Net latency: %.2f s" % tracker.latency,
                        org=(20, height - 40),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 0, 255))
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

    # Release everything if job is finished
    out.release()
    cap.release()
    cv2.destroyAllWindows()
