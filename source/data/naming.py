# THIS CLASS EXISTS BECAUSE PATH RESOLUTION IS A MESS
import os


def __robust_respath_search():
    curpath = os.path.realpath(__file__)
    basepath = curpath
    while os.path.split(basepath)[1] != 'source':
        newpath = os.path.split(basepath)[0]
        if newpath == basepath:
            print("ERROR: unable to find source from path " + curpath)
            break
        basepath = os.path.split(basepath)[0]
    return os.path.join(os.path.split(basepath)[0], "resources")


# ######### RESOURCES DIRECTORIES DEFINITION ###########

RESPATH = __robust_respath_search()
TBFOLDER = "tbdata"
MODELSFOLDER = "models"
CROPPERSFOLDER = os.path.join(MODELSFOLDER, "croppers")
JLOCATORSFOLDER = os.path.join(MODELSFOLDER, "jlocators")
DATASETSFOLDER = "datasets"
CROPSDATAFOLDER = os.path.join(DATASETSFOLDER, "crops")
JOINTSDATAFOLDER = os.path.join(DATASETSFOLDER, "joints")

# #################### DATASET-COMPONENT DEFINES ###############

TRAIN_IN = 'TRAIN_IN'
TRAIN_TARGET = 'TRAIN_TARGET'
TRAIN_TARGET2 = 'TRAIN_TARGET2'
VALID_IN = 'VALID_IN'
VALID_TARGET = 'VALID_TARGET'
VALID_TARGET2 = 'VALID_TARGET2'

CROPPER = 'cropper'
JLOCATOR = 'jlocator'
RAND = 'RAND'
SEQUENTIAL = 'SEQ'


def resources_path(*paths):
    p = os.path.join(RESPATH, *paths)
    if os.path.splitext(p)[1] != '':
        basep = os.path.split(p)[0]
    else:
        basep = p
    os.makedirs(basep, exist_ok=True)
    return p

# ############################## BASE DIRECTORY-RELATIVE PATHS ###############


def tensorboard_path(*paths):
    return resources_path(TBFOLDER, *paths)


def models_path(*paths):
    return resources_path(MODELSFOLDER, *paths)


def croppers_path(*paths):
    return resources_path(CROPPERSFOLDER, *paths)


def joint_locators_path(*paths):
    return resources_path(JLOCATORSFOLDER, *paths)


def datasets_path(*paths):
    return resources_path(DATASETSFOLDER, *paths)


def crops_path(*paths):
    return resources_path(CROPSDATAFOLDER, *paths)


def joints_path(*paths):
    return resources_path(JOINTSDATAFOLDER, *paths)

# ######################### MODEL NAME CONVENTIONS ###################


def cropper_h5_path(name):
    return croppers_path(name+".h5")


def cropper_ckp_path(name):
    return croppers_path(name+".ckp")


def jlocator_h5_path(name):
    return joint_locators_path(name+".h5")


def jlocator_ckp_path(name):
    return joint_locators_path(name+".ckp")

# ############################## LEGACY #########################################


class PathManager:
    """
    This class automatically finds the project root folder at instantiation. It is used
    to resolve the resources folder regardless of the current running path.
    """
    def __init__(self):
        self.__res_path = RESPATH

    def resources_path(self, path):
        return os.path.join(self.__res_path, path)


def list_files(basedir):
    """
    Provide a list of all paths of all files recursively included in some of the
    base directories provided in basedir
    :param basedir: An arbitrarily nested list of files and folders to consider
    :return: a list of paths of files included in some directory in the input list
    """
    rets = []
    if not isinstance(basedir, str):
        for d in basedir:
            rets.extend(list_files(d))
        return rets

    if not os.path.isdir(basedir):
        return [basedir]

    subelems = [os.path.join(basedir, f) for f in os.listdir(basedir)]
    for f in subelems:
        rets.extend(list_files(f))
    return rets


if __name__ == '__main__':
    print(joints_path("j1", "jj.mat"))
    print(croppers_path("mod1.ckp"))
    print(joint_locators_path("j1", "jj1"))
    print(crops_path())

