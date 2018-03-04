import numpy as np
# Here we define a formatting convention for raw data into semantics.

WRIST = 'WRIST'
THUMB = 'THUMB'
INDEX = 'INDEX'
MIDDLE = 'MIDDLE'
RING = 'RING'
BABY = 'BABY'

FINGERS = (THUMB, INDEX, MIDDLE, RING, BABY)
ALL_AREAS = (WRIST, THUMB, INDEX, MIDDLE, RING, BABY)


def hand_format(raw):
    """
    Standard definition of the formatted representation of hand-related data
    with regards to joints
    :param raw: the raw data as a sequence of joint data in the standard order
    :return: a dictionary with name-indexed arrays of joints
    """
    return {
        WRIST: [raw[0]],
        THUMB: __finger_from(raw, 1),
        INDEX: __finger_from(raw, 5),
        MIDDLE: __finger_from(raw, 9),
        RING: __finger_from(raw, 13),
        BABY: __finger_from(raw, 17)
    }


def raw(form):
    """
    From dictionary-formatted data get back to the raw array representation
    :param form: dictionary-formatted data
    :return: the raw array of all joints in standard order
    """
    return np.concatenate((form[WRIST],
                           form[THUMB],
                           form[INDEX],
                           form[MIDDLE],
                           form[RING],
                           form[BABY]))


def get_label_and_index(rawidx):
    if rawidx == 0:
        return WRIST, 0
    if rawidx <= 4:
        return THUMB, rawidx - 1
    if rawidx <= 8:
        return INDEX, rawidx - 5
    if rawidx <= 12:
        return MIDDLE, rawidx - 9
    if rawidx <= 16:
        return RING, rawidx - 13
    return BABY, rawidx - 17


def __finger_from(raw, index):
    return [raw[index], raw[index+1], raw[index+2], raw[index+3]]
