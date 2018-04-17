from library.geometry.calibration import *
from data.datasets.io.hand_io import *
from library.geometry.formatting import *


def build_default_hand_model():
    SCALE_FACTOR = 1
    img, raw_positions = load("prototype.mat")

    raw_positions = [(x * img.shape[0], y * img.shape[1]) for (x, y, f) in raw_positions]
    cal = calibration(intr=synth_intrinsic(resolution=img.shape[0:2], fov=(15, 15 * img.shape[0] / img.shape[1])))
    points = np.array([ImagePoint(elem, depth=1).to_camera_model(
        calibration=cal).as_row()
                       for elem in raw_positions * SCALE_FACTOR])
    return hand_format(points - np.average(points, axis=0))
