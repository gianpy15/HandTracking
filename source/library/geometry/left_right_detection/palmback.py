from library.geometry.formatting import *

RIGHT = 1
LEFT = -1

# coordinate system defines
YDXR = -1   # Y-down,   X-right (framedata default)
YUXR = 1    # Y-up,     X-right
YDXL = 1    # Y-down,   X-left
YUXL = -1   # Y-up,     X-left


def leftright_to_palmback(hand: dict, side: int, coordinate_system=YDXR, delta=0.3):
    """
    Determine whether the given hand is visible by palm (+1.0) or back (-1.0)
    :param hand: the hand joints in rich format in image coordinates
    :param side: the side of the hand, use defines LEFT or RIGHT
    :param coordinate_system: the coordinate system of the image, default is Y-down, X-right
    :param delta: the confidence bound for uncertainty over which side (palm/back) is visible
    :return: a real number in [-1.0, 1.0] indicating the confidence of:
        palm visible (+1.0)
        back visible (-1.0)
    """
    v_index = hand[INDEX][0] - hand[WRIST][0]
    v_baby = hand[BABY][0] - hand[WRIST][0]
    v_index = v_index / np.linalg.norm(v_index[0:2])
    v_baby = v_baby / np.linalg.norm(v_baby[0:2])
    s = (v_index[0] * v_baby[1] - v_index[1] * v_baby[0]) * side * coordinate_system
    if s > delta:
        return 1.
    elif s < -delta:
        return -1.
    return s / delta


if __name__ == '__main__':
    from data.datasets.framedata_management.naming import *
    from data.datasets.io.hand_io import *
    from matplotlib.pyplot import imshow, show
    fname = "handsBorgo3"
    index = 10 * 7 + 3
    rgb, labels = load(get_complete_frame_path(frame_name(fname, index)), format=(RGB_DATA, LABEL_DATA))
    hand_data = hand_format(labels)
    imshow(rgb)
    show()
    print(leftright_to_palmback(hand=hand_data, side=LEFT))