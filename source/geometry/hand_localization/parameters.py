from geometry.formatting import *

# Maximum rotation angles of any joint around the median rotation axis
maxangle = {
    THUMB: (np.pi / 180 * 45, np.pi / 180 * 55, np.pi / 180 * 65),
    INDEX: (np.pi / 180 * 50, np.pi / 180 * 45, np.pi / 180 * 45),
    MIDDLE: (np.pi / 180 * 45, np.pi / 180 * 45, np.pi / 180 * 45),
    RING: (np.pi / 180 * 45, np.pi / 180 * 45, np.pi / 180 * 45),
    BABY: (np.pi / 180 * 47, np.pi / 180 * 45, np.pi / 180 * 45)
}

# Maximum rotation angles of any joint far from the main plain
maxwideangle = {
    THUMB: (np.pi / 180 * 10, np.pi / 180 * 2, np.pi / 180 * 0),
    INDEX: (np.pi / 180 * 20, np.pi / 180 * 2, np.pi / 180 * 0),
    MIDDLE: (np.pi / 180 * 10, np.pi / 180 * 2, np.pi / 180 * 0),
    RING: (np.pi / 180 * 15, np.pi / 180 * 2, np.pi / 180 * 0),
    BABY: (np.pi / 180 * 20, np.pi / 180 * 5, np.pi / 180 * 0)
}

# The angles defining where is the median rotation axis of the finger.
# Median rotation is the one pointing at this angle in between the
# palm and the finger direction
normdirangles = {
    THUMB: np.pi / 180 * 45,
    INDEX: np.pi / 180 * 45,
    MIDDLE: np.pi / 180 * 45,
    RING: np.pi / 180 * 45,
    BABY: np.pi / 180 * 45
}

START_DIR = 'sd'
AROUND_DIR = 'ad'
NORM_DIR = 'nd'
MAXANGLE = 'ma'
MAXWIDEANGLE = 'mwa'
THUMBAXIS = 'thax'

LEFT = -1
RIGHT = 1

# Number of fast finger estimates to find out what direction is the palm
SIDE_N_ESTIM = 3
# Fingers used to estimate the direction of the hand
DIRECTION_REVEALING_FINGERS = FINGERS

normdirrates = {}
for finger in FINGERS:
    normdirrates[finger] = {
        START_DIR: np.sin(normdirangles[finger]) ** 2,
        NORM_DIR: np.cos(normdirangles[finger]) ** 2
    }
