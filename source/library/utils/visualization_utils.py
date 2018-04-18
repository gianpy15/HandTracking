import numpy as np
from skimage.transform import resize


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
            images.append((k + (1 - k) * tmp_mask) * image[i])
        return np.array(images)

    if np.shape(image)[:-1] != np.shape(mask)[:-1]:
        mask = resize(mask, output_shape=np.shape(image)[:-1])
    return (k + ((1 - k) * mask)) * image
