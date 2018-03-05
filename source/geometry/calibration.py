import numpy as np

# camera calibration and image-model mapping
# for a reference about calibration see:
# https://it.mathworks.com/help/vision/ug/camera-calibration.html?s_tid=gn_loc_drop
# https://prateekvjoshi.com/2014/05/31/understanding-camera-calibration/

# the following formula is considered:
# [X, Y, 1] = [X, Y, Z, 1] * EXTRINSIC_MAT * INTRINSIC_MAT / Z

# INTRINSICS are the camera parameters that define how the world is projected into the image
INTRINSIC = 'int'
# EXTRINSICS are the parameters that define the world position of the camera
EXTRINSIC = 'ext'
DEPTHSCALE = 'dsca'  # the conversion from depth sensor units to world units
FOCAL_X = 'fx'  # FOCAL_LENGTH/PIXEL_X_LENGTH in world units
FOCAL_Y = 'fy'  # FOCAL_LENGTH/PIXEL_Y_LENGTH in world units
CENTER_X = 'cx'  # IMAGE CENTER X IN PIXELS (may not be the middle pixel)
CENTER_Y = 'cy'  # IMAGE CENTER Y IN PIXELS (may not be the middle pixel)
SKEW = 'sk'  # FOCAL_Y * tan(alpha) where alpha is the skew angle of the pixel

# default value intrinsics describe 600x600 perfectly centered images, no skew, cubic camera
default_intrinsic = {
    FOCAL_X: 600.0,
    FOCAL_Y: 600.0,
    CENTER_X: 300.0,
    CENTER_Y: 300.0,
    SKEW: 0
}

# default value describe the model seen from the camera perspective
default_extrinsic = np.array([[1, 0, 0],  # ROTATION
                              [0, 1, 0],  # ROTATION
                              [0, 0, 1],  # ROTATION
                              [0, 0, 0]])  # TRANSLATION

# default value measures the depth directly in world units
default_depth_scale = 1

default_calibration = {
    INTRINSIC: default_intrinsic,
    EXTRINSIC: default_extrinsic,
    DEPTHSCALE: default_depth_scale
}

# Set up this to globally change the calibration used
current_calibration = default_calibration


def set_current_calib(cal):
    global current_calibration
    current_calibration = cal


def intrinsic_matrix(calibration):
    """
    Extract the intrinsic matrix transformation from the calibration settings
    :param calibration: the calibration dictionary to consider
    :return: a numpy array representing the intrinsic matrix transformation
    """
    if INTRINSIC in calibration.keys():
        conf = calibration[INTRINSIC]
    else:
        conf = calibration
    return np.array([[conf[FOCAL_X], 0, 0],
                     [conf[SKEW], conf[FOCAL_Y], 0],
                     [conf[CENTER_X], conf[CENTER_Y], 1]])


def model2image(modelpoint, calibration, makedepth=False):
    image_point = np.matmul(np.matmul(modelpoint.as_row_tr(), calibration[EXTRINSIC]),
                            intrinsic_matrix(calibration[INTRINSIC]))

    if image_point[2] != 0:
        image_point = image_point[0:2] / image_point[2]
    else:
        image_point = np.zeros(shape=(2,))
    ret = ImagePoint(image_point[0:2])
    if not makedepth:
        return ret
    else:
        ret.depth = __depth_from_model(modelpoint, calibration)
        return ret


def depthimage2cameramodel(imagepoint, calibration):
    if imagepoint.depth is None:
        return image2cameramodeldirectin(imagepoint, calibration)
    z = imagepoint.depth / calibration[DEPTHSCALE]
    x = (imagepoint.coords[0] - calibration[INTRINSIC][CENTER_X]) * z / calibration[INTRINSIC][FOCAL_X]
    y = (imagepoint.coords[1] - calibration[INTRINSIC][CENTER_Y]) * z / calibration[INTRINSIC][FOCAL_Y]
    return ModelPoint((x, y, z))


def image2cameramodeldirectin(imagepoint, calibration):
    sample = ImagePoint(imagepoint.coords, depth=calibration[DEPTHSCALE])
    sample = depthimage2cameramodel(sample, calibration)
    sample /= np.linalg.norm(sample.as_row(), ord=2)
    return sample


def synth_intrinsic(resolution, fov):
    """
    Synthesize an intrinsic dictionary configuration based on desired resolution and field of view.
    The given intrinsics will have the image center in the middle and no skew.
    :param resolution: a tuple describing the number of pixels for each dimension: (pixx, pixy)
    :param fov: a tuple describing the desired field of view in degrees for each axis: (hor_fov, vert_fov)
    :return: a dictionary containing the desired intrinsic configuration
    """
    intr = {
        SKEW: 0,
        CENTER_X: resolution[0] / 2,
        CENTER_Y: resolution[1] / 2
    }
    radian_half_fov = np.array(fov) * np.pi / 360.0
    focal = np.array(resolution) / (2 * np.tan(radian_half_fov))
    intr[FOCAL_X] = focal[0]
    intr[FOCAL_Y] = focal[1]
    return intr


def calibration(intr=default_intrinsic, ext=default_extrinsic, depthscale=default_depth_scale):
    return {
        INTRINSIC: intr,
        EXTRINSIC: ext,
        DEPTHSCALE: depthscale
    }


ZR300_RES = (640, 480)
ZR300_FOV = (68, 41.5)
ZR300_CAL = calibration(synth_intrinsic(resolution=ZR300_RES,
                                        fov=ZR300_FOV))


def __depth_from_model(modelpoint, calibration):
    camera_model_point = np.matmul(modelpoint.as_row_tr(), calibration[EXTRINSIC])
    return calibration[DEPTHSCALE] * camera_model_point[2]


class ImagePoint:
    def __init__(self, coords, depth=None):
        self.coords = coords[0:2]
        self.depth = depth if depth != 0 else None
        self.visible = True if self.depth is not None else False

    def to_camera_model(self, calibration=None):
        if calibration is None:
            calibration = current_calibration
        return depthimage2cameramodel(self, calibration=calibration)

    def as_row(self):
        return np.array(self.coords[0:2])

    def as_col(self):
        return np.transpose([self.coords[0:2]])

    def as_row_tr(self):
        return np.concatenate((self.coords[0:2], [1]))

    def as_col_tr(self):
        return np.transpose([self.as_row_tr()])


class ModelPoint:
    def __init__(self, coords):
        self.coords = coords[0:3]

    def as_row(self):
        return np.array(self.coords[0:3])

    def as_col(self):
        return np.transpose([self.coords[0:3]])

    def as_row_tr(self):
        return np.concatenate((self.coords[0:3], [1]))

    def as_col_tr(self):
        return np.transpose([self.as_row_tr()])

    def to_image_space(self, calibration=None, makedepth=False):
        if calibration is None:
            calibration = current_calibration
        return model2image(self, calibration, makedepth=makedepth)

    def __truediv__(self, other):
        return ModelPoint(self.coords/other)

    def __itruediv__(self, other):
        self.coords /= other
        return self

    def __sub__(self, other):
        return ModelPoint(np.array(self.coords)-np.array(other.coords))
