import numpy as np


def __heatmap_dequantize(heat):
    return heat / 255.0


CROPIMGFORMAT = ('frame', lambda x: x)
CROPHEATMAPFORMAT = ('heatmap', __heatmap_dequantize)
CROPDEPTHFORMAT = ('depth', lambda x: np.reshape(x, newshape=np.shape(x)+(1,)))

JUNCIMGFORMAT = ('cut', lambda x: x)
JUNCHEATFORMAT = ('heatmap_array', __heatmap_dequantize)
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