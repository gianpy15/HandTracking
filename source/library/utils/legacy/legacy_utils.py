import tensorflow as tf
import sys
import numpy as np


def get_index(array):
    """
    Get the index of the argmax over a collection. No clue why this function exists.
    :param array: The array where you are supposed to extract the argmax
    :return: The index of the argmax
    NB: writing np.argmax(array) is faster and more clear. Do this way.
    """
    return np.argmax(array)


def completion(n):
    """
    Print fancy completion incremental bar! [===>---------]
    :param n: should ask to @gianpy15, but should be the current progress over 100
    :return: the fancy string of the incremental progress bar!
    """
    c = list("[")
    if n >= 5:
        c.append('>')
        n = n - 5

    for i in range(19):
        if n >= 5:
            if c[-1] == '>':
                c[-1] = '='
            c.append('>')
        else:
            c.append('-')
        n = n - 5
    c.append(']')
    return "".join(c)


def accuracy(exp_out, out, name='accuracy', merge_summary_group=None):
    if merge_summary_group is not None:
        name_1 = merge_summary_group.current_scope
    else:
        name_1 = name

    with tf.name_scope(name_1):
        correct_prediction = tf.equal(tf.argmax(exp_out, 1), tf.argmax(out, 1))
        acc_1 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        if merge_summary_group is not None:
            merge_summary_group.add_scalar(acc_1, name)
        return acc_1


def one_hot_coding(in_num):
    base = [0] * 10
    base[in_num[0] - 1] = 1
    return base


def get_formatted_out(outputs):
    return [one_hot_coding(num) for num in outputs]


def __scale_range__(to_scale, current_max, desired_max):
    """
    rescaling utility for integers or lists of integers from 0 to current_max

    note: the rescaling is done in order to project the [0, current_max] interval
    uniformly into the 0 .. desired_max integer interval
    :param to_scale: the value/list of values to be scaled
    :param current_max: the current cap of the [0, current_max] interval
    :param desired_max: the cap of the 0 .. desired_max integer interval
    :return: the integer value in 0 .. desired_max corresponding to the input
    """
    epsilon = 1e-5
    return np.uint32(np.array(np.float_(to_scale)) * (float(desired_max) / (current_max + epsilon)))


def __get_greyscale_colcode__(pixel, elem_max=255):
    """
    get the greyscale color encoding

    greyscale encoding is: 232 + GREYVALUE
    where GREYVALUE must be an integer between 0 and 23

    the reason to this is to be complementary with the RGB encoding that takes
    values from 16 to 231

    :param pixel: the GREYSCALE pixel to be encoded
    :param elem_max: the maximum value it is supposed to take, for rescaling purposes
    :return: the string encoding the pixel's color
    """
    code = str(232 + __scale_range__(pixel, elem_max, 23))
    return code


def get_rgb_colcode(pixel, elem_max=255):
    """
    get the RGB color encoding

    RGB encoding is: 16 + 36*RED + 6*GREEN + BLUE

    where RED, GREEN and BLUE must be integers between 0 and 5

    :param pixel: the RGB pixel to be encoded
    :param elem_max: the maximum value it is supposed to take, for rescaling purposes
    :return: the string encoding the pixel's color
    """
    scaled_pixel = __scale_range__(pixel, elem_max, 5)
    return str(16 + 36 * scaled_pixel[0] + 6 * scaled_pixel[1] + scaled_pixel[2])


def get_print_symbol(pixel, elem_max=255, symbols=(' ', '.', 'c', 'c', 'o', 'O', 'G', '0', '#'), colors=True):
    """
    Decode the element into a string to print
    :param pixel: the pixel to be printed, either greyscale or RGB
    :param elem_max: the maximum value it is supposed to take, for rescaling purposes
    :param symbols: the alphabet of strings that could be printed, in magnitude order
    :param colors: whether to print colors in escape codes or not
    :return: the string encoding the pixel
    """
    pre_text = "\033[38;5;"
    pre_back = "\033[48;5;"
    if len(pixel) == 1:
        col_code = __get_greyscale_colcode__(pixel[0], elem_max=elem_max)
        value = pixel[0]
    else:
        col_code = get_rgb_colcode(pixel[0:3], elem_max=elem_max)
        value = pixel[0] / 3 + pixel[1] / 3 + pixel[2] / 3
    col_terminator = "m"
    col_reset = "\033[0m"
    level = __scale_range__(value, elem_max, len(symbols))
    if colors:
        return pre_text + col_code + col_terminator + pre_back + col_code + col_terminator + '  ' + col_reset
    return symbols[level]


def print_data_images(img_batch, colors=True):
    """
    Print a batch of images on the stdout

    :param img_batch: the image batch to be printed, 4 dimensional
    :param colors: boolean: whether to use colors or not
    """
    for img in img_batch:
        max_value = 0.0
        min_value = 1000.0
        # find out the maximum value of the image in order to
        # rescale colors
        for row in img:
            for pixel in row:
                for channel in pixel:
                    if channel > max_value:
                        max_value = channel
                    if channel < min_value:
                        min_value = channel
        # now print the image pixel per pixel
        for row in img:
            for pixel in row:
                sys.stdout.write(get_print_symbol(pixel - min_value, elem_max=max_value - min_value, colors=colors))
            sys.stdout.write('\n')
        sys.stdout.write('\n')


def rescale_img(img, min=0.0, max=1.0):
    img_min = img[0, 0, 0]
    img_max = img[0, 0, 0]
    for row in img:
        for pixel in row:
            for channel in pixel:
                if channel > img_max:
                    img_max = channel
                elif channel < img_min:
                    img_min = channel
    range = img_max - img_min
    out_img = np.array(
        [[[min + (channel - img_min) / range * max for channel in pixel] for pixel in row] for row in img])
    return out_img


def downsample_img(imgs, sample_jump=2):
    img_shape = np.shape(imgs)
    if len(img_shape) == 3:
        rows = img_shape[0]
        cols = img_shape[1]
        out_img = np.array([
            [imgs[r, p, :] for p in range(cols) if p % sample_jump == 0]
            for r in range(rows) if r % sample_jump == 0])
    else:
        rows = img_shape[1]
        cols = img_shape[2]
        out_img = np.array([[
            [single_image[r, p, :] for p in range(cols) if p % sample_jump == 0]
            for r in range(rows) if r % sample_jump == 0]
            for single_image in imgs])
    return out_img


def apply_pixelwise_transform(imgs, transform, **kwargs):
    out_img = np.array([[
        [transform(pixel, **kwargs) for pixel in row]
        for row in img]
        for img in imgs
    ])
    return out_img


def make_sprite(img_batch, force_square=False):
    batch_dims = img_batch.shape
    w = batch_dims[2]
    h = batch_dims[1]
    b_size = batch_dims[0]
    total_pixels = w * h * b_size
    imgs_per_row = np.ceil(np.sqrt(total_pixels) / w)
    imgs_per_col = np.ceil(b_size / imgs_per_row)

    if force_square:
        imgs_per_row = imgs_per_col = max(imgs_per_col, imgs_per_row)
    if imgs_per_col * imgs_per_row < b_size:
        print("ERROR! this algorithm is broken.")

    imgs_per_row = np.int(imgs_per_row)
    imgs_per_col = np.int(imgs_per_col)
    out_matrix = np.zeros(shape=[imgs_per_col * h, imgs_per_row * w, 3])

    for row in range(len(out_matrix)):
        for col in range(len(out_matrix[row])):
            batch_index = np.int32(np.floor(col / w) + imgs_per_row * np.floor(row / h))
            if batch_index < b_size:
                local_row = row % h
                local_col = col % w
                out_matrix[row, col] = img_batch[batch_index, local_row, local_col]

    return out_matrix


def mark_images(imgs, labels, label_to_rgb_dict, border_size=4):
    out_imgs = imgs[:, :, :, :]
    for i in range(len(imgs)):
        for row in range(len(imgs[i])):
            for col in range(len(imgs[i, row])):
                if row < border_size or row > len(imgs[i]) - border_size \
                        or col < border_size or col > len(imgs[i, row]) - border_size:
                    out_imgs[i, row, col] = label_to_rgb_dict[labels[i][0]]

    return out_imgs
