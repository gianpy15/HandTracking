# Here we define a formatting convention for raw data into semantics.

WRIST = 'w'
THUMB = 't'
INDEX = 'i'
MIDDLE = 'm'
RING = 'r'
BABY = 'b'


def hand_format(raw):
    return {
        WRIST: (raw[0],),
        THUMB: __finger_from(raw, 1),
        INDEX: __finger_from(raw, 5),
        MIDDLE: __finger_from(raw, 9),
        RING: __finger_from(raw, 13),
        BABY: __finger_from(raw, 17)
    }


def __finger_from(raw, index):
    return raw[index], raw[index+1], raw[index+2], raw[index+3]
