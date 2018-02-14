# THIS CLASS EXISTS BECAUSE PATH RESOLUTION IS A MESS
# PLEASE MANUALLY SET THIS MODE DEPENDING ON HOW YOU ARE RUNNING YOUR STUFF

# modes available:
# CONSOLE - executing from source directory from console
# IDE - executing from pycharm
import os


class PathManager:
    def __init__(self):
        curpath = os.path.realpath(".")
        basepath = curpath
        while basepath.split('/')[-1] != 'source':
            newpath = os.path.split(basepath)[0]
            if newpath == basepath:
                print("ERROR: unable to find source from path "+curpath)
                break
            basepath = os.path.split(basepath)[0]
        self.__res_path = os.path.join(os.path.split(basepath)[0], "resources")

    def resources_path(self, path):
        return os.path.join(self.__res_path, path)


def list_files(basedir):
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
