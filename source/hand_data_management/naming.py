from image_loader.hand_io import pm
import os


framebase = pm.resources_path("framedata")
contributors = pm.resources_path("framedata/contributors.txt")
TEMPDIR = "tmp"


def frame_name(vidname, frameno):
    return "%s%04d.mat" % (vidname, frameno)


def index_name(vidname):
    return "%s-index.txt" % (vidname,)


def cached_frame_name(vidname, frameno):
    return "%s%04d.png" % (vidname, frameno)


def get_frameno(filename):
    return int(filename[-8:-4])


def get_vidname(filename):
    return filename.split("/")[-1][0:-8]


def get_index_from_vidname(vidname):
    viddir = os.path.join(framebase, vidname)
    return os.path.join(viddir, index_name(vidname))


def get_index_from_frame(framename):
    return get_index_from_vidname(get_vidname(framename))


def get_complete_frame_path(framename):
    vidname = get_vidname(framename)
    return os.path.join(os.path.join(framebase, vidname), framename)


def get_tmp_dir_from_vidname(vidname):
    viddir = os.path.join(framebase, vidname)
    return os.path.join(viddir, TEMPDIR)

