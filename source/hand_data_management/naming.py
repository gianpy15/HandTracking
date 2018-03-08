from data_manager import path_manager as pm
import os


framebase = pm.resources_path("framedata")
contributors = os.path.join(framebase, "contributors.txt")
TEMPDIR = "tmp"
NUMDIGITS = 4
NUMREPR = "%0"+str(NUMDIGITS)+"d"
FRAME_NAME_BASE = "%s"+NUMREPR


def frame_name(vidname, frameno):
    return (FRAME_NAME_BASE+".mat") % (vidname, frameno)


def index_name(vidname):
    return "%s-index.txt" % (vidname,)


def cached_frame_name(vidname, frameno):
    return (FRAME_NAME_BASE+".png") % (vidname, frameno)


def get_frameno(filename):
    return int(os.path.splitext(filename)[0][-NUMDIGITS:])


def get_vidname(filename):
    return os.path.splitext(os.path.split(filename)[1])[0][0:-NUMDIGITS]


def get_index_from_vidname(vidname):
    viddir = os.path.join(framebase, vidname)
    return os.path.join(viddir, index_name(vidname))


def get_index_from_frame(framename):
    return get_index_from_vidname(get_vidname(framename))


def get_vid_dir_from_vidname(vidname):
    return os.path.join(framebase, vidname)


def get_complete_frame_path(framename):
    vidname = get_vidname(framename)
    return os.path.join(os.path.join(framebase, vidname), framename)


def get_tmp_dir_from_vidname(vidname):
    return os.path.join(framebase, vidname, TEMPDIR)
