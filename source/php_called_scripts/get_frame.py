import sys
import os
sys.path.append(os.path.realpath(os.path.join(os.path.split(__file__)[0], "..")))

from php_called_scripts.utils import *

if __name__ == '__main__':
    img, name = select_best_overall_frame()
    imgstr = encode_image(img)
    print(name + SEPARATOR + imgstr)
    set_index_flag(get_index_from_frame(name),
                   flag=FLAG_PROCESSING,
                   idx=get_index_from_frame(name))
