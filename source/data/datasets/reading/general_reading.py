import scipy.io as scio
from data.naming import *


def read_frame(path: str, *, format):
    if not os.path.isfile(path):
        tpath = resources_path(path)
        if not os.path.isfile(path):
            raise IOError("Unable to find file.\nTried paths:\n%s\n%s" % (path, tpath))
        path = tpath

    data = scio.loadmat(path)
    res = []
    for (matkey, consumer) in format:
        res.append(consumer(data[matkey]))
    return res


def read_frame_batch(frames: list, *, format):
    frame_count = len(frames)
    assert frame_count > 0
    f1 = read_frame(frames[0], format=format)
    ret = [None for _ in range(len(f1))]

    for idx in range(len(f1)):
        ret[idx] = np.empty(shape=(frame_count,)+np.shape(f1[idx]), dtype=f1[idx].dtype)
        ret[idx][0] = f1[idx]

    for frameidx in range(1, frame_count):
        fn = read_frame(frames[frameidx], format=format)
        for compidx in range(len(fn)):
            ret[compidx][frameidx] = fn[compidx]

    return ret


def read_formatted_batch(frames: list, formatdict: dict):
    mapping_dict = {}
    offset_counter = 0
    outformat = []
    for k in formatdict.keys():
        mapping_dict[k] = []
        for elem in formatdict[k]:
            mapping_dict[k].append(offset_counter)
            outformat.append(elem)
            offset_counter += 1
    data_elements = read_frame_batch(frames, format=outformat)
    out = {}
    for k in mapping_dict.keys():
        elemlist = [data_elements[i] for i in mapping_dict[k]]
        out[k] = np.concatenate(elemlist, axis=-1)

    return out


if __name__ == '__main__':
    from matplotlib import pyplot as pplt
    from data.datasets.reading.formatting import *

    set_verbosity(DEBUG)
    frames = [crops_path("handsDondoo1_137.mat"), crops_path("handsBorgo2_1.mat")]
    dictformat = {
        'in': (CROPIMGFORMAT, CROPDEPTHFORMAT),
        'out': (CROPHEATMAPFORMAT,)
    }

    data = read_formatted_batch(frames, formatdict=dictformat)
    pplt.imshow(data['in'][0, :, :, 0:3])
    pplt.show()
    pplt.imshow(data['out'][0, :, :])
    pplt.show()