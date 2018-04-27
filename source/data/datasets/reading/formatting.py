import numpy as np
from data.naming import IN, OUT


CROPIMGFORMAT = ['frame', lambda x: x / 255.0]
CROPHEATMAPFORMAT = ['heatmap', lambda x: np.expand_dims(x / 255.0, axis=-1)]
CROPDEPTHFORMAT = ['depth', lambda x: np.expand_dims(x, axis=-1)]

JUNCIMGFORMAT = ['cut', lambda x: x / 255.0]
JUNCHEATFORMAT = ['heatmap_array', lambda x: np.expand_dims(x / 255.0, axis=-1)]
JUNCVISFORMAT = ['visible', lambda x: x]


def format_set_field_name(name, format, entry_index, channel_index=0):
    format[entry_index][channel_index][0] = name
    return format

def format_add_inner_func(f, format, entry_index, channel_index=0):
    old_f = format[entry_index][channel_index][1]
    format[entry_index][channel_index][1] = lambda x: old_f(f(x))
    return format

def format_add_outer_func(f, format, entry_index, channel_index=0):
    old_f = format[entry_index][channel_index][1]
    format[entry_index][channel_index][1] = lambda x: f(old_f(x))
    return format


CROPS_STD_FORMAT = {
    IN(0): [CROPIMGFORMAT],
    OUT(0): [CROPHEATMAPFORMAT]
}

CROPS_STD_DEPTH_FORMAT = {
    IN(0): [CROPIMGFORMAT, CROPDEPTHFORMAT],
    OUT(0): [CROPHEATMAPFORMAT]
}

JUNC_STD_FORMAT = {
    IN(0): [JUNCIMGFORMAT],
    OUT(0): [JUNCHEATFORMAT],
    OUT(1): [JUNCVISFORMAT]
}