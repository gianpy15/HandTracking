# THIS CLASS EXISTS BECAUSE PATH RESOLUTION IS A MESS

# unused imports are intended to unify the naming files
# about data naming conventions and defines

# if any new naming path convention is needed, just:
#   1) Notify the group
#   2) Add it here
#       2.1) Add its RESOURCES DIRECTORY DEFINITION (currently around line 35)
#       2.2) Add its BASE DIRECTORY-RELATIVE PATH (currently around line 200)
#       2.3) Optionally add MODEL NAME CONVENTIONS (currently around line 270)
#   3) Use everywhere the path functions you just added and expect everybody to do the same
import os
import re
from library.utils.deprecation import deprecated_class

def __robust_respath_search():
    """
    Resolve the path for resources from anywhere in the code.
    :return: The real path of the resources
    """
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
PALMBACKFOLDER = os.path.join(DATASETSFOLDER, "palmback")
JSONHANDSFOLDER = os.path.join(DATASETSFOLDER, "jsonhands")

# #################### DATASET-COMPONENT DEFINES ###############

# ### DEPRECATED ###
TRAIN_IN = 'TRAIN_IN'
TRAIN_TARGET = 'TRAIN_TARGET'
TRAIN_TARGET2 = 'TRAIN_TARGET2'
VALID_IN = 'VALID_IN'
VALID_TARGET = 'VALID_TARGET'
VALID_TARGET2 = 'VALID_TARGET2'
GENERIC_IN = 'GENERIC_IN'
GENERIC_TARGET = 'GENERIC_TARGET'
GENERIC_TARGET2 = 'GENERIC_TARGET2'


class NameGenerator:
    """
    This class generates names with a fixed pattern to be able to identify
    back at any moment the names it generated.
    Useful to standardize cross-code dependencies (ex: model definition to training dictionary)
    """
    def __init__(self, prefix, separator):
        self.prefix = prefix
        self.separator = separator
        self.regex = re.compile("^%s%s(?P<key>.*)" % (self.prefix, self.separator))

    def __call__(self, iden):
        """
        Produce a name from the fixed pattern
        :param iden: any kind of identifier. If int, pads up to 4 digits to enable sorting.
        :return: a name uniquely identified by iden, following the pattern from self
        """
        if isinstance(iden, int):
            return "%s%s%04d" % (self.prefix, self.separator, iden)
        return "%s%s%s" % (self.prefix, self.separator, iden)

    def __getitem__(self, item):
        return self(item)

    def filter(self, names):
        return [name for name in names if re.match(pattern=self.regex,
                                                   string=name)]

    def reverse(self, keys):
        if isinstance(keys, str):
            match = self.regex.match(keys)
            if match is None:
                raise KeyError("Unable to reverse key %s produced by a different NameGenerator. My keys have form: %s"
                               % (keys, self.regex))
            return match.groups("key")[0]
        ret = []
        for key in keys:
            r = self.reverse(key)
            if isinstance(r, str):
                ret.append(r)
            else:
                ret += r
        return ret


# network input-output conventions

# This NameGenerator should be used to specify all inputs in a Model
# you can use any kind of identifier to name the component
# ex:
#
# inputs = keras.layers.Input(input_shape=(None, None, 3), name=IN(0))
# side_inputs = keras.Input(input_shape=(21,), name=IN(1))
#
# At the end they must be the inputs of the final Model.
# All of them must be named with IN
IN = NameGenerator(prefix='IN',
                   separator='_')

# This NameGenerator should be used to specify all outputs in a Model
# you can use any kind of identifier to name the component
# ex:
#
# out = keras.layers.Conv2D(filters=1,
#                           kernel_size=[1, 1],
#                           activation='sigmoid',
#                           name=OUT(0))(act7)
# secondary_output = keras.layers.Dense(units=1,
#                                       activation='sigmoid',
#                                       use_bias=True,
#                                       name=OUT('pb_class')(x)
#
# At the end they must be the outputs of the final Model.
# All of them must be named with OUT
OUT = NameGenerator(prefix='OUT',
                    separator='_')

# This NameGenerator is intended to refer to the runtime outputs of a Model
# Use it to refer to runtime dynamic computations in contrapposition with OUT
# that holds the corresponding target outputs.
NET_OUT = NameGenerator(prefix='NETOUT',
                        separator='_')

CROPPER = 'cropper'
JLOCATOR = 'jlocator'
RAND = 'RAND'
SEQUENTIAL = 'SEQ'

def resources_path(*paths):
    """
    Very base function for resources path management.
    Return the complete path from resources given a sequence of directories
    eventually terminated by a file, and makes all necessary subdirectories
    :param paths: a sequence of paths to be joined starting from the base of resources
    :return: the complete path from resources (all necessary directories are created)
    """
    p = os.path.join(RESPATH, *paths)
    if os.path.splitext(p)[1] != '':
        basep = os.path.split(p)[0]
    else:
        basep = p
    os.makedirs(basep, exist_ok=True)
    return p

# ############################## BASE DIRECTORY-RELATIVE PATHS ###############


def tensorboard_path(*paths):
    """
    Builds the path starting where all tensorboard data should be.
    :param paths: sequence of directories to be joined after the standard base.
    :return: The path relative to this standard folder
    """
    return resources_path(TBFOLDER, *paths)


def models_path(*paths):
    """
    Builds the path starting where all model data should be.
    :param paths: sequence of directories to be joined after the standard base.
    :return: The path relative to this standard folder
    """
    return resources_path(MODELSFOLDER, *paths)


def croppers_path(*paths):
    """
    Builds the path starting where all cropper models data should be.
    :param paths: sequence of directories to be joined after the standard base.
    :return: The path relative to this standard folder
    """
    return resources_path(CROPPERSFOLDER, *paths)


def joint_locators_path(*paths):
    """
    Builds the path starting where all joint locator models data should be.
    :param paths: sequence of directories to be joined after the standard base.
    :return: The path relative to this standard folder
    """
    return resources_path(JLOCATORSFOLDER, *paths)


def datasets_path(*paths):
    """
    Builds the path starting where all datasets should be.
    :param paths: sequence of directories to be joined after the standard base.
    :return: The path relative to this standard folder
    """
    return resources_path(DATASETSFOLDER, *paths)


def crops_path(*paths):
    """
    Builds the path starting where all datasets about crops should be.
    :param paths: sequence of directories to be joined after the standard base.
    :return: The path relative to this standard folder
    """
    return resources_path(CROPSDATAFOLDER, *paths)


def joints_path(*paths):
    """
    Builds the path starting where all datasets about joint locations should be.
    :param paths: sequence of directories to be joined after the standard base.
    :return: The path relative to this standard folder
    """
    return resources_path(JOINTSDATAFOLDER, *paths)


def palmback_path(*paths):
    """
    Builds the path starting where all datasets about palm/back classificators should be.
    :param paths: sequence of directories to be joined after the standard base.
    :return: The path relative to this standard folder
    """
    return resources_path(PALMBACKFOLDER, *paths)


def jsonhands_path(*paths):
    """
    Builds the path starting where all datasets about palm/back classificators should be.
    :param paths: sequence of directories to be joined after the standard base.
    :return: The path relative to this standard folder
    """
    return resources_path(JSONHANDSFOLDER, *paths)

# ######################### MODEL NAME CONVENTIONS ###################


def cropper_h5_path(name):
    """
    Builds the standard full path of a .H5 cropper model given its base name
    :param name: the base name identifier of the H5 model
    :return: The standard path of the relative .H5 file.
    """
    return croppers_path(name+".h5")


def cropper_ckp_path(name):
    """
    Builds the standard full path of a .ckp cropper model given its base name
    :param name: the base name identifier of the ckp model
    :return: The standard path of the relative .ckp file.
    """
    return croppers_path(name+".ckp")


def jlocator_h5_path(name):
    """
    Builds the standard full path of a .H5 joint locator model given its base name
    :param name: the base name identifier of the H5 model
    :return: The standard path of the relative .H5 file.
    """
    return joint_locators_path(name+".h5")


def jlocator_ckp_path(name):
    """
    Builds the standard full path of a .ckp joint locator model given its base name
    :param name: the base name identifier of the ckp model
    :return: The standard path of the relative .ckp file.
    """
    return joint_locators_path(name+".ckp")

# ############################## LEGACY #########################################
# This stuff was designed ages ago... please don't use it


@deprecated_class(alternative=resources_path)
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
    for i in range(10):
        print(IN(i+0.))
        print(OUT(i))

    print(IN.reverse(IN.filter(['92', 'IN_IN_IN', 'IN_', 'IN_23', 'OUT_O', 'aIN_a'])))

