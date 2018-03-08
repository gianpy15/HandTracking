from data_manager import path_manager as pm
import scipy.io as scio


RGB_DATA = 'data'
LABEL_DATA = 'labels'
DEPTH_DATA = 'depth'

ALL_DATA = (RGB_DATA, LABEL_DATA, DEPTH_DATA)


def load(respath, format=(RGB_DATA, LABEL_DATA)):
    """
    Load data from a .mat file implementing all HandTracking naming conventions
    :param respath: the path of the .mat file taken from the resources base directory
    :param format: specify the intended format of the returned tuple
    :return: a tuple of numpy arrays in the order specified by param format,
            fields are None if not present
    """
    matdict = scio.loadmat(pm.resources_path(respath))

    def retrieve_content(tag):
        if tag in matdict.keys():
            return matdict[tag]
        return None

    ret = []
    for tag in format:
        ret.append(retrieve_content(tag))
    return tuple(ret)


def store(respath, data=None, labels=None, depth=None):
    """
    Store data on a .mat file implementing all HandTracking naming conventions
    :param respath: the path of the .mat file taken from the resources base directory
    :param data: the optional frame data to be stored as a numpy matrix
    :param labels: the optional labels to be associated with the frame
    :param depth: the optional depth map to be associated with the frame
    """
    outdict = {}
    if data is not None:
        outdict[RGB_DATA] = data
    if labels is not None:
        outdict[LABEL_DATA] = labels
    if depth is not None:
        outdict[DEPTH_DATA] = depth
    if len(outdict.keys()) == 0:
        return

    respath = pm.resources_path(respath)
    scio.savemat(respath, outdict)