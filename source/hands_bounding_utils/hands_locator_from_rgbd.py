import source.hand_data_management.camera_data_conversion as cdc
import source.hand_data_management.video_loader as vl
import source.hand_data_management.grey_to_redblue_codec as gtrbc
import numpy as np
from scipy.ndimage import convolve
import hands_bounding_utils.utils as u
from data_manager.path_manager import PathManager
pm = PathManager()


def read_mesh_video(result_name, framedir):
    resp = pm.resources_path("vid1.mp4")
    frames = pm.resources_path("rawcam/out-1520009971")
    cdc.read_mesh_video(resp, frames)


def load_labelled_videos(vname, verbosity = 0):
    frames, labels = vl.load_labeled_video(vname)
    frames = np.array(frames)
    labels = np.array(labels)
    if verbosity == 1:
        print("FRAMES SHAPE: ", frames.shape)
        print("LABELS SHAPE: ", labels.shape)
    return frames, labels


def load_all_frames_data(framespath, verbosity = 0):
    framespathreal = pm.resources_path(framespath)
    frames = cdc.read_frame_data(**cdc.default_read_rgb_args(framespathreal))
    depths = cdc.read_frame_data(**cdc.default_read_z16_args(framespathreal))
    frames = np.array(frames)
    depths = np.array(depths)
    if verbosity == 1:
        print("FRAMES SHAPE: ", frames.shape)
        print("LABELS SHAPE: ", depths.shape)
    return frames, depths


def get_numbered_frame(framespath, number, verbosity=0):
    frames1, depths1 = load_all_frames_data(framespath, verbosity)
    nframe = frames1[number]
    ndepth = depths1[number]
    return nframe, ndepth


def single_depth_frame_to_redblue(depthframe):
    depthsfirstframe = gtrbc.codec(np.array([depthframe]))
    return depthsfirstframe[0]


def show_frame(frame):
    u.showimage(frame)


def __right_derivative_kernel():
    return np.array([[-1, 1]])


def __left_derivative_kernel():
    return np.array([[1, -1]])


def __ones_kernel(dim):
    return np.ones([dim, dim])


def eliminate_too_far(depth, toofartheshold):
    depth[depth > toofartheshold] = 0
    return depth


def eliminate_too_close(depth, tooclosetheshold):
    depth[depth < tooclosetheshold] = 0
    return depth


def normalize_non_zeros(depth):
    depth[depth != 0] = 1
    return depth


def elementwise_product(frame, map):
    frame1 = np.multiply(frame[:, :, 0], map)
    frame2 = np.multiply(frame[:, :, 1], map)
    frame3 = np.multiply(frame[:, :, 2], map)
    frame_rec = np.dstack((frame1, frame2))
    frame_rec = np.dstack((frame_rec, frame3))
    return np.uint8(frame_rec)


def transorm_rgd_depth(frame, depth, showimages = False, toofar=1500, tooclose=500):
    depth1 = np.squeeze(depth)
    if showimages:
        show_frame(depth1)
    depth1 = eliminate_too_far(depth1, toofar)
    depth1 = eliminate_too_close(depth1, tooclose)
    depth1 = normalize_non_zeros(depth1)
    if showimages:
        show_frame(depth1)
    depth1 = convolve(depth1, __ones_kernel(15))
    depth1 = normalize_non_zeros(depth1)
    frame1 = elementwise_product(frame, depth1)
    if showimages:
        show_frame(frame1)
        show_frame(depth1)
    return frame1, depth1


firstframe, firstdepth = get_numbered_frame("rawcam/out-1520009971", 214)


# firstdepthtoshow = single_depth_frame_to_redblue(firstdepth)
# show_frame(firstdepthtoshow)

firstframe, firstdepth = transorm_rgd_depth(firstframe, firstdepth,showimages=True)






