import numpy as np


CROPIMGFORMAT = ('frame', lambda x: x / 255.0)
CROPHEATMAPFORMAT = ('heatmap', lambda x: x / 255.0)
CROPDEPTHFORMAT = ('depth', lambda x: np.reshape(x, newshape=np.shape(x)+(1,)))

JUNCIMGFORMAT = ('cut', lambda x: x / 255.0)
JUNCHEATFORMAT = ('heatmap_array', lambda x: x / 255.0)
JUNCVISFORMAT = ('visible', lambda x: x)


IN = 'in'
TARGET = 'target'
SEC_TARGET = 'sec_target'

CROPS_STD_FORMAT = {
    IN: (CROPIMGFORMAT,),
    TARGET: (CROPHEATMAPFORMAT,)
}

CROPS_STD_DEPTH_FORMAT = {
    IN: (CROPIMGFORMAT, CROPDEPTHFORMAT),
    TARGET: (CROPHEATMAPFORMAT,)
}

JUNC_STD_FORMAT = {
    IN: (JUNCIMGFORMAT,),
    TARGET: (JUNCHEATFORMAT,),
    SEC_TARGET: (JUNCVISFORMAT,)
}