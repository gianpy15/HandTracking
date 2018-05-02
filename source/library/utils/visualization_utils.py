import numpy as np
from skimage.transform import resize
from data import *
from data.datasets.crop.utils import get_crops_from_heatmap


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
def crop_sprite(feed, sprite_shape=(300, 300, 3)):
    img = feed[IN(0)]
    heat = feed[NET_OUT(0)]
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
            if np.shape(image[i])[:-1] != np.shape(mask[i])[:-1]:
                tmp_mask = resize(mask[i], output_shape=np.shape(image[i])[:-1])
            images.append((k + (1 - k) * tmp_mask) * (image[i] - np.min(image[i])))
        return np.array(images)

    if np.shape(image)[:-1] != np.shape(mask)[:-1]:
        mask = resize(mask, output_shape=np.shape(image)[:-1])
    return (k + ((1 - k) * mask)) * (image - np.min(image))
