import numpy as np
import timeit as time
from scipy import io as scio
from os import path as os_p
from skimage import io
import matplotlib.pyplot as plt
import math


# ################### READING FILES ###########################
from tensorflow import device


def read_image(path):
    """opens the image at the given path and returns its numpy version"""
    return np.array(io.imread(path))


def __read_mat_hand_bounding(path, structasrecord=True):
    """opens & reads the .mat file at the specified path and returns the present structures"""
    return np.squeeze(np.array(scio.loadmat(path, struct_as_record=structasrecord)['boxes']))


def __get_points_list_from_boxes_structures_hand_mat(structures):
    """returns an array of shape (4,2,n) where n=len(structures) excluding
    eventual "None" values present in structures"""
    coord_sets = []
    for struct in structures:
        if struct is not None:
            mat = []
            for i in range(4):
                mat.append(np.squeeze(np.array(struct[0][0][i])))
            coord_sets.append(mat)
    return np.array(coord_sets)


def __read_coords_from_mat_file(path):
    """returns an array of shape (4,2,n) where n is the number of structures saved in the .mat file whose path
     is given as parameter, excluding eventual "None" values present in structures"""
    return __get_points_list_from_boxes_structures_hand_mat(__read_mat_hand_bounding(path))


# ###################### CROPS #####################à


def __get_bounds(coord):
    """given an array of 4 coordinates (x,y), simply computes and
    returns the highest and lowest vertical and horizontal points"""
    if len(coord) != 4:
        raise AttributeError("coord must be a set of 4 coordinates")
    x = [coord[i][0] for i in range(len(coord))]
    y = [coord[i][1] for i in range(len(coord))]
    up = coord[np.argmin(np.array(x))][0]
    down = coord[np.argmax(np.array(x))][0]
    left = coord[np.argmin(np.array(y))][1]
    right = coord[np.argmax(np.array(y))][1]
    return up, down, left, right


def __copy_area(image, up, down, left, right):
    """copies from image, the specified area"""
    up = int(up)
    down = int(down)
    left = int(left)
    right = int(right)
    return image[up:down, left:right]


def cropimage(imagepath, matfilepath, save=False, enlarge=0.3):
    """returs the crop(s) of the image located at imagepath, with the coordinates taken from the .mat file
    located at matfilepath
    :type imagepath: string representing the path of the image to crop
    :type matfilepath: string representing the path of the .mat file containing the coordinates of the crops
    :type save: set true to save the crops as .jpg
    :type enlarge: must be non-negative, crop are enlarged by the given percentage. Default is 0,3 (30%)"""
    if enlarge < 0:
        raise AttributeError("enlarge must be non-negative")
    coords = __read_coords_from_mat_file(matfilepath)
    image = read_image(imagepath)
    crops = []
    for coord in coords:
        cropped_image = __crop_from_coords(image, coord, enlarge)
        crops.append(cropped_image)
    if save:
        split = os_p.splitext(imagepath)
        for i in range(0, len(crops)):
            io.imsave(split[0]+"_crop"+str(i)+split[1], crops[i])
    return np.array(crops)


def __crop_from_coords(image, coord, enlarge):
    """given a list of 4 coordinates (coord) and an image, returns a crop of the image w.r.t the given coordinates,
    enlarged by a factor defined by the enlarge parameter."""
    image_height = len(image)
    image_width = len(image[0])
    up, down, left, right = __get_bounds(coord)
    up -= (down - up) * (enlarge / 2)
    down += (down - up) * (enlarge / 2)
    left -= (right - left) * (enlarge / 2)
    right += (right - left) * (enlarge / 2)
    if up < 0:
        up = 0
    if left < 0:
        left = 0
    if right > image_width:
        right = image_width
    if down > image_height:
        down = image_height
    cropped_image = __copy_area(image, up, down, left, right)
    return cropped_image


def __get_coords_from_heatmap(heatmap, precision, height_shrink_rate, width_shrink_rate,
                              accept_crop_minimum_dimension_pixels):
    """given a heatmap, returns a set of set of coordinates representing bounds of the crops that have to be done.
    To generate crops, all the areas of the heatmap that contain points with values >= precision
    are taken into account."""
    coords = []
    bounds = []
    coords_not = []
    bounds_not = []
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            if heatmap[i][j] >= precision and not __contains(bounds, i, j) and not __contains(bounds_not, i, j):
                coord = __get_biggest_connected_area_from_point(heatmap, precision, [[i, j]])
                up, down, left, right = __get_bounds(coord)
                if (down-up+1)*height_shrink_rate*(right-left+1)*width_shrink_rate \
                        >= accept_crop_minimum_dimension_pixels:
                    coords.append(coord)
                    bounds.append([up, down, left, right])
                else:
                    coords_not.append(coord)
                    bounds_not.append([up, down, left, right])

    return np.array(coords)


def __get_biggest_connected_area_from_point(heatmap, precision, lista):
    in_list = np.zeros(heatmap.shape)
    in_list[lista[0][0]][lista[0][1]] = 1
    for p in lista:
            if p[0] > 0 and heatmap[p[0]-1][p[1]] >= precision and in_list[p[0]-1][p[1]] != 1:
                lista.append([p[0]-1, p[1]])
                in_list[p[0] - 1][p[1]] = 1
            if p[0] < heatmap.shape[0]-1 and heatmap[p[0]+1][p[1]] >= precision and in_list[p[0]+1][p[1]] != 1:
                lista.append([p[0]+1, p[1]])
                in_list[p[0] + 1][p[1]] = 1
            if p[1] < heatmap.shape[1]-1 and heatmap[p[0]][p[1]+1] >= precision and in_list[p[0]][p[1]+1] != 1:
                lista.append([p[0], p[1]+1])
                in_list[p[0]][p[1]+1] = 1
            if p[1] > 0 and heatmap[p[0]][p[1]-1] >= precision and in_list[p[0]][p[1]-1] != 1:
                lista.append([p[0], p[1]-1])
                in_list[p[0]][p[1] - 1] = 1
    list_x = np.array([p[0] for p in lista])
    list_y = np.array([p[1] for p in lista])
    min_x = np.min(list_x)
    max_x = np.max(list_x)
    min_y = np.min(list_y)
    max_y = np.max(list_y)
    return [[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]


def __contains(bounds, point_x, point_y):
    """returns True if at least one of the bounds of the list bounds contains the given point.
    returns False otherwise"""
    for bound in bounds:
        if bound[0] <= point_x <= bound[1] and bound[2] <= point_y <= bound[3]:
            return True
    return False


def __resize_coords(coords, height_shrink_rate, width_shrink_rate):
    """this function serves the get_crops_from_heatmap function.
    Since heatmaps have a scaled dimension w.r.t. the original image, this function computes the inverse
    re-scaling, converting the heatmap coordinates to image coordinates, and returns the new coordinates set"""
    coords_new = coords*[height_shrink_rate, width_shrink_rate]
    return coords_new


def get_crops_from_heatmap(image, heatmap, height_shrink_rate=10, width_shrink_rate=10, precision=0.5,
                           enlarge=0.5, accept_crop_minimum_dimension_pixels=1000):
    """given an image path and a heatmap, returns an array of images representing the crops of the image w.r.t.
    the given heatmap.
    :type image: the image to crop
    :type heatmap: a matrix containing all values in the range [0,1] that represents where hands are most likely
    present in the given image.
    :type enlarge: must be non-negative, crops are enlarged by the given percentage. Default is 0,3 (30%)
    :type precision: represents which values of the heatmap will be taken into account for the crops
    :type height_shrink_rate: the ratio used to rescale from the image height to heatmap height
    :type width_shrink_rate: the ratio used to rescale from the image width to heatmap width
    :type accept_crop_minimum_dimension_pixels: due to noise is may be possible that single pixels or small areas
    will be detected as possible crops. All crops that are smaller than this parameter (square_pixels) are deleted
    The default value for this parameter is 1000px,"""
    coords = __get_coords_from_heatmap(heatmap, precision, height_shrink_rate, width_shrink_rate,
                                       accept_crop_minimum_dimension_pixels)
    coords = __resize_coords(coords, height_shrink_rate, width_shrink_rate)
    crops = []
    for coord in coords:
        cropped_image = __crop_from_coords(image, coord, enlarge)
        crops.append(cropped_image)
    return crops


# ############################## PRODUCING HEATMAP ###########################

def __get_containment_bounds(up, down, left, right, container_up, container_down, container_left, container_right):
    """given 2 rectangles defined by (up, down, left, right) and
    (container_up, container_down, container_left, container_right), returns 4 integers representing the
    overlapping area. Returns (0, 0, 0, 0) if they don't overlap."""
    ris_up = -1
    ris_down = -1
    ris_left = -1
    ris_right = -1
    if container_up < up < container_down:
        ris_up = up
    if container_up < down < container_down:
        ris_down = down
    if container_left < left < container_right:
        ris_left = left
    if container_left < right < container_right:
        ris_right = right
    if __count_minus_one(ris_up, ris_down, ris_left, ris_right) >= 3:
        return 0, 0, 0, 0
    if ris_right == ris_left == -1:
        return 0, 0, 0, 0
    if ris_up == ris_down == -1:
        return 0, 0, 0, 0
    if ris_up == -1:
        ris_up = container_up
    if ris_down == -1:
        ris_down = container_down
    if ris_left == -1:
        ris_left = container_left
    if ris_right == -1:
        ris_right = container_right
    return ris_up, ris_down, ris_left, ris_right


def __count_minus_one(a, b, c, d):
    """serves __get_containment_bounds ONLY. Counts, among the 4 input elements, how many of them are equal to -1"""
    acc = 0
    if a == -1:
        acc += 1
    if b == -1:
        acc += 1
    if c == -1:
        acc += 1
    if d == -1:
        acc += 1
    return acc


def heatmap_to_rgb(heat):
    """converts a bi-dimensional heatmap, whose values are in [0,1] to a rgb image"""
    heat2 = np.zeros((heat.shape[0], heat.shape[1], 3))
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            heat2[i][j] = np.float64(np.array([255, 255, 255]))*heat[i][j]
    return np.uint8(heat2)


def get_heatmap_from_mat(image, matfilepath, heigth_shrink_rate=10, width_shrink_rate=10, overlapping_penalty=0.9):
    """given an image and a .mat file containing coordinates of the rectangles where hands are present,
    returns a heatmap that, according to high values, defines where hands are present. Note that the heatmap
    is re-sized w.r.t the original image size. The shrink rates are defined by height_shrink_rate and width_shrink_rate.
    The just mentioned parameters can be tuned according to the precision that is required for the application the
    heatmap will be used in."""
    if heigth_shrink_rate < 1:
        raise AttributeError("height_shrink_rate should be greated than 1")
    if width_shrink_rate < 1:
        raise AttributeError("width_shrink_rate should be greated than 1")
    coords = __read_coords_from_mat_file(matfilepath)
    image_height = len(image)
    image_width = len(image[0])
    heatmap_height = int(math.ceil(image_height / heigth_shrink_rate))
    heatmap_width = int(math.ceil(image_width / width_shrink_rate))
    heatmap = np.zeros((heatmap_height, heatmap_width))
    for coord in coords:
        up, down, left, right = __get_bounds(coord)
        for i in range(heatmap_height):
            for j in range(heatmap_width):
                c_up, c_down, c_left, c_right = __get_containment_bounds(i*heigth_shrink_rate,
                                                                         (i+1)*heigth_shrink_rate,
                                                                         j*width_shrink_rate,
                                                                         (j+1)*width_shrink_rate,
                                                                         up, down, left, right)
                if not (c_up == c_down == c_left == c_right == 0):
                    if heatmap[i][j] == 0:
                        heatmap[i][j] = ((c_down-c_up)*(c_right-c_left))/(heigth_shrink_rate*width_shrink_rate)
                    else:
                        heatmap[i][j] = (1-overlapping_penalty)*(((c_down-c_up)*(c_right-c_left)) /
                                                                 (heigth_shrink_rate*width_shrink_rate)+heatmap[i][j])
    heatmap = heatmap / np.max(heatmap)
    return heatmap


# ############# UTILS ##########################

def showimages(images):
    """being images an array of images, displays them all"""
    for image in images:
        showimage(image)


def showimage(image):
    """displays a single image"""
    plt.figure()
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    imagep = "dataset/images/Poselet_186.jpg"
    matp = "dataset/annotations/Poselet_186.mat"
    image1 = read_image(imagep)
    # showimages(cropimage(imagep, matp))
    heatmap1 = get_heatmap_from_mat(image1, matp)
    showimage(heatmap_to_rgb(heatmap1))
    showimages(get_crops_from_heatmap(image1, heatmap1, enlarge=0.2))


    def timetest():
        get_crops_from_heatmap(image1, heatmap1, precision=0.7)


    # print(time.timeit(stmt=timetest, number=1))
