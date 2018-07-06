import numpy as np
from skimage.transform import resize
from data import *
from data.datasets.crop.utils import get_crops_from_heatmap
from data.datasets.jlocator.heatmaps_to_hand import heatmaps_to_hand
from library.geometry.formatting import *
from skimage.draw import *


def hex_to_rgb(hexcode: str):
    hexcode = hexcode.strip('#')
    return np.array([int(hexcode[i:i+2], 16)/255 for i in (0, 2, 4)])


PALMCOL = hex_to_rgb('#000000')

THUMBCOL1 = hex_to_rgb('#285E1C')
THUMBCOL2 = hex_to_rgb('#388E3C')
THUMBCOL3 = hex_to_rgb('#58AE4C')

INDEXCOL1 = hex_to_rgb('#95AF21')
INDEXCOL2 = hex_to_rgb('#DEEF41')
INDEXCOL3 = hex_to_rgb('#FEFF71')

MIDDLECOL1 = hex_to_rgb('#750000')
MIDDLECOL2 = hex_to_rgb('#A50000')
MIDDLECOL3 = hex_to_rgb('#F50000')

RINGCOL1 = hex_to_rgb('#4B0072')
RINGCOL2 = hex_to_rgb('#7B1FA2')
RINGCOL3 = hex_to_rgb('#9B2FFF')

BABYCOL1 = hex_to_rgb('#003560')
BABYCOL2 = hex_to_rgb('#1565C0')
BABYCOL3 = hex_to_rgb('#3585F0')


SEGMENTS_LIST = {
    # WRIST
    0: [],
    # THUMB
    1: [[0, PALMCOL]],
    2: [[1, THUMBCOL1]],
    3: [[2, THUMBCOL2]],
    4: [[3, THUMBCOL3]],
    # INDEX
    5: [[0, PALMCOL], [1, PALMCOL]],
    6: [[5, INDEXCOL1]],
    7: [[6, INDEXCOL2]],
    8: [[7, INDEXCOL3]],
    # MIDDLE
    9: [[0, PALMCOL], [5, PALMCOL]],
    10: [[9, MIDDLECOL1]],
    11: [[10, MIDDLECOL2]],
    12: [[11, MIDDLECOL3]],
    # RING
    13: [[0, PALMCOL], [9, PALMCOL]],
    14: [[13, RINGCOL1]],
    15: [[14, RINGCOL2]],
    16: [[15, RINGCOL3]],
    # BABY
    17: [[0, PALMCOL], [13, PALMCOL]],
    18: [[17, BABYCOL1]],
    19: [[18, BABYCOL2]],
    20: [[19, BABYCOL3]]
}


def batch_tb_logging(fun):
    """
    Decorator to turn a image-wise computation of image tb logging
    into a complete batch-wise computation.
    NOTE: doing batch computation by hand may result in being much more memory efficient.
    :param fun: the image-wise function to decorate
    :return: the batch-wise equivalent
    """
    def wrap(feed, *args, **kwargs):
        batchlen = len(feed[list(feed.keys())[0]])
        res = []
        for idx in range(batchlen):
            single_feed = {}
            for k in feed:
                single_feed[k] = feed[k][idx]
            res.append(fun(single_feed, *args, **kwargs))
        return np.array(res)
    return wrap


@batch_tb_logging
def crop_sprite(feed, sprite_shape=(300, 300, 3), img_key=IN(0), heat_key=NET_OUT(0)):
    img = feed[img_key]
    heat = feed[heat_key]
    height_sr = np.shape(img)[0] / np.shape(heat)[0]
    width_sr = np.shape(img)[1] / np.shape(heat)[1]
    pixcount = np.shape(img)[0] * np.shape(img)[1]
    crops = get_crops_from_heatmap(image=img,
                                   heatmap=heat,
                                   height_shrink_rate=height_sr,
                                   width_shrink_rate=width_sr,
                                   accept_crop_minimum_dimension_pixels=pixcount/100)
    ret = np.zeros(shape=sprite_shape, dtype=np.float32)

    def attach_crop(start, end, crop):
        cropshape = np.shape(crop)
        height_ratio = cropshape[0] / (end[0]-start[0])
        width_ratio = cropshape[1] / (end[1]-start[1])
        ratio = max(height_ratio, width_ratio)
        output_shape = int(cropshape[0] / ratio), int(cropshape[1] / ratio)
        crop = resize(crop, output_shape=output_shape)
        ret[start[0]:start[0]+output_shape[0], start[1]:start[1]+output_shape[1], :] = crop
        ret[start[0], :, 0] = 1.
        ret[:, start[1], 0] = 1.
        ret[end[0]-1, :, 0] = 1.
        ret[:, end[1]-1, 0] = 1.

    if len(crops) == 0:
        return ret
    if len(crops) == 2:
        crop0_shape = np.shape(crops[0])
        crop1_shape = np.shape(crops[1])
        ratio0 = crop0_shape[0] / crop0_shape[1]
        ratio1 = crop1_shape[0] / crop1_shape[1]
        comparable0 = ratio0 if ratio0 > 1 else 1 / (ratio0 + 1e-5)
        comparable1 = ratio1 if ratio1 > 1 else 1 / (ratio1 + 1e-5)
        if comparable0 > comparable1:
            ratio = ratio0
        else:
            ratio = ratio1
        if ratio > 1:
            attach_crop(start=(0, 0),
                        end=(sprite_shape[0], int(sprite_shape[1]//2)),
                        crop=crops[0])
            attach_crop(start=(0, int(sprite_shape[1]//2)),
                        end=(sprite_shape[0], sprite_shape[1]),
                        crop=crops[1])
        else:
            attach_crop(start=(0, 0),
                        end=(int(sprite_shape[0]//2), sprite_shape[1]),
                        crop=crops[0])
            attach_crop(start=(int(sprite_shape[0] // 2), 0),
                        end=(sprite_shape[0], sprite_shape[1]),
                        crop=crops[1])
        return ret
    gridsize = int(np.ceil(np.sqrt(len(crops))))
    delta = (int(sprite_shape[0] / gridsize), int(sprite_shape[1] / gridsize))
    for idx1 in range(gridsize):
        for idx2 in range(gridsize):
            if len(crops)-1 < idx1 * gridsize + idx2:
                break
            start = (delta[0] * idx1, delta[1] * idx2)
            attach_crop(start=start,
                        end=(start[0]+delta[0], start[1]+delta[1]),
                        crop=crops[idx1 * gridsize + idx2])
    return ret


def get_image_with_mask(image, mask, k=0.15):
    """
    Given an image or a batch of images, and the same number
    of heat maps, this function return the images with the
    mask in transparency depending on the value of k as follows
    result = (k + ((1-k) * mask)) * image
    :param image: is the batch of images
    :param mask: is the heatmap to apply
    :param k: is the transparency of the heatmap w.r.t. to the image
    :return: a batch of images or a single image
    """
    if len(np.shape(image)) == 4:
        images = []

        for i in range(len(image)):
            tmp_mask = mask[i]
            im_min = np.min(image[i])
            im_max = np.max(image[i])
            if np.shape(image[i])[:-1] != np.shape(mask[i])[:-1]:
                tmp_mask = resize(mask[i], output_shape=np.shape(image[i])[:-1])
            images.append(255*(k + (1 - k) * tmp_mask) * (image[i] - im_min) / (im_max-im_min))
        return np.array(images)

    im_min = np.min(image)
    im_max = np.max(image)
    if np.shape(image)[:-1] != np.shape(mask)[:-1]:
        mask = resize(mask, output_shape=np.shape(image)[:-1])
    return 255*(k + ((1 - k) * mask)) * (image - im_min) / (im_max-im_min)


@batch_tb_logging
def joint_skeleton_impression(feed, img_key=IN(0),
                              heats_key=NET_OUT(0),
                              vis_key=NET_OUT(1)):
    img = feed[img_key]
    imgmin = np.min(img)
    imgmax = np.max(img)
    img = (img - imgmin)/(imgmax-imgmin)
    heats = feed[heats_key]
    vis = feed[vis_key]
    hand = heatmaps_to_hand(joints=heats,
                            visibility=vis)
    hand = raw(hand)
    out = np.array(img)
    outshape = np.shape(out)
    green = np.array([0, 1, 0])
    blue = np.array([0, 0, 1])
    for idx in range(len(hand)):
        joint = hand[idx]
        coords = int(joint[0] * outshape[0]), int(joint[1] * outshape[1])
        for (idx2, col) in SEGMENTS_LIST[idx]:
            joint2 = hand[idx2]
            coords2 = int(joint2[0] * outshape[0]), int(joint2[1] * outshape[1])
            rr, cc, val = line_aa(coords[0], coords[1],
                                  coords2[0], coords2[1])
            out[rr, cc] = np.array([col * v + out[r, c] * (1-v) for (r, c, v) in zip(rr, cc, val)])
        col = joint[2] * green + (1-joint[2]) * blue
        cir = circle(coords[0], coords[1], 2, shape=outshape)
        out[cir] = col
    return out


if __name__ == '__main__':
    from matplotlib import pyplot as mplt
    dm = DatasetManager(train_samples=1,
                        valid_samples=1,
                        batch_size=1,
                        dataset_dir=joints_path(),
                        formatting=JUNC_STD_FORMAT)
    data = dm.train()

    plain = data[0][IN(0)][0]
    print(np.shape(plain))
    mplt.imshow(plain)
    mplt.show()
    skeletal = joint_skeleton_impression(feed=data[0],
                                         heats_key=OUT(0),
                                         vis_key=OUT(1))
    print(np.shape(skeletal))
    mplt.imshow(skeletal[0])
    mplt.show()