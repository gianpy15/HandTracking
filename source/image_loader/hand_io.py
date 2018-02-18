from data_manager import path_manager
import scipy.io as scio

pm = path_manager.PathManager()

DATA_F = 'data'
LABEL_F = 'labels'


def load(respath):
    matdict = scio.loadmat(pm.resources_path(respath))
    if LABEL_F in matdict.keys():
        return matdict[DATA_F], matdict[LABEL_F]
    return matdict[DATA_F]


def store(respath, data, labels=None):
    if labels is None:
        scio.savemat(respath, {DATA_F: data})
    else:
        scio.savemat(respath, {DATA_F: data, LABEL_F: labels})