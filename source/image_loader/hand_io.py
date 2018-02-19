from data_manager import path_manager
import scipy.io as scio

pm = path_manager.PathManager()

DATA_F = 'data'
LABEL_F = 'labels'


def load(respath):
    """
    Load data from a .mat file implementing all HandTracking naming conventions
    :param respath: the path of the .mat file taken from the resources base directory
    :return: a tuple of numpy arrays in the order (data, labels), labels is None if not present
    """
    matdict = scio.loadmat(pm.resources_path(respath))
    if LABEL_F in matdict.keys():
        return matdict[DATA_F], matdict[LABEL_F]
    return matdict[DATA_F], None


def store(respath, data, labels=None):
    """
    Store data on a .mat file implementing all HandTracking naming conventions
    :param respath: the path of the .mat file taken from the resources base directory
    :param data: the frame data to be stored as a numpy matrix
    :param labels: the optional labels to be associated with the data
    """
    respath = pm.resources_path(respath)
    if labels is None:
        scio.savemat(respath, {DATA_F: data})
    else:
        scio.savemat(respath, {DATA_F: data, LABEL_F: labels})