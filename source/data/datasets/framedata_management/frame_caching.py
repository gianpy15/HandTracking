from hand_data_management.naming import *
from image_loader.image_loader import save_image_from_matrix
import image_loader.hand_io as hio
import os


def cache_frame(frame):
    vidname = get_vidname(frame)
    frameno = get_frameno(frame)
    tmpdir = get_tmp_dir_from_vidname(vidname)
    cached_frame = os.path.join(tmpdir, cached_frame_name(vidname, frameno))
    if os.path.isfile(cached_frame):
        return cached_frame
    framedata, _ = hio.load(get_complete_frame_path(frame))
    save_image_from_matrix(framedata, cached_frame)
    return cached_frame


def uncache_frame(frame):
    vidname = get_vidname(frame)
    frameno = get_frameno(frame)
    tmpdir = get_tmp_dir_from_vidname(vidname)
    cached_frame = os.path.join(tmpdir, cached_frame_name(vidname, frameno))
    if not os.path.isfile(cached_frame):
        return False
    os.remove(cached_frame)
    return True
