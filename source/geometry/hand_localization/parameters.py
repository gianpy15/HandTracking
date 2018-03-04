from geometry.formatting import *


# Definitions for the configuration of the finger-building process:
START_DIR = 'sd'        # The initial versor of the finger before any transformation
NORM_DIR = 'nd'         # The versor pointing towards the rotation side.
AROUND_DIR = 'ad'       # The versor around which the main angle must be measured. Lin. dep. on START and NORM dirs
MAXANGLE = 'ma'         # The maximum angle allowed around the AROUND_DIR
MAXWIDEANGLE = 'mwa'    # The maximum angle distance allowed from the plane formed by START_DIR and NORM_DIR
THUMBAXIS = 'thax'      # For the thumb, say around which axis it is really rotating to correct its self rotation


# Maximum rotation angles of any joint around the AROUND_DIR
maxangle = {
    THUMB: (np.pi / 180 * 45, np.pi / 180 * 55, np.pi / 180 * 65),
    INDEX: (np.pi / 180 * 50, np.pi / 180 * 45, np.pi / 180 * 45),
    MIDDLE: (np.pi / 180 * 45, np.pi / 180 * 45, np.pi / 180 * 45),
    RING: (np.pi / 180 * 45, np.pi / 180 * 45, np.pi / 180 * 45),
    BABY: (np.pi / 180 * 47, np.pi / 180 * 45, np.pi / 180 * 45)
}

# Maximum rotation angles of any joint far from the main plain (built by START_DIR, NORM_DIR)
maxwideangle = {
    THUMB: (np.pi / 180 * 10, np.pi / 180 * 10, np.pi / 180 * 0.1),
    INDEX: (np.pi / 180 * 20, np.pi / 180 * 20, np.pi / 180 * 0.1),
    MIDDLE: (np.pi / 180 * 10, np.pi / 180 * 20, np.pi / 180 * 0.1),
    RING: (np.pi / 180 * 15, np.pi / 180 * 20, np.pi / 180 * 0.1),
    BABY: (np.pi / 180 * 20, np.pi / 180 * 50, np.pi / 180 * 0.1)
}

# The angles defining where is the AROUND_DIR of the finger.
# The AROUND_DIR is built pointing at this angle in between the
# NORM_DIR and the START_DIR versors
normdirangles = {
    THUMB: np.pi / 180 * 45,
    INDEX: np.pi / 180 * 45,
    MIDDLE: np.pi / 180 * 45,
    RING: np.pi / 180 * 45,
    BABY: np.pi / 180 * 45
}

# Compute the actual rates to give to the two START and NORM versors to compute the AROUND versor
normdirrates = {}
for finger in FINGERS:
    normdirrates[finger] = {
        START_DIR: np.sin(normdirangles[finger]) ** 2,
        NORM_DIR: np.cos(normdirangles[finger]) ** 2
    }

# Definitions for the LEFT-RIGHT detection. VALUES ARE SIGNIFICANT, DO NOT MODIFY.
LEFT = -1
RIGHT = 1

# Number of fast finger estimates to find out what side is the palm
SIDE_N_ESTIM = 3
# Fingers used to estimate the side of the hand
DIRECTION_REVEALING_FINGERS = FINGERS
