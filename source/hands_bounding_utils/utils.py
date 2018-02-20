import numpy as np
from scipy import io as scio
from os import path as os_p
from skimage import io
import matplotlib.pyplot as plt


def __read_image(path):
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


def __get_bounds(coord):
    """given an array of 4 coordinates (x,y), simply computes and
    returns the highest and lowest vertical and horizontal points"""
    if len(coord)!= 4:
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
    ris = []
    up = int(up)
    down = int(down)
    left = int(left)
    right = int(right)
    for i in range(up, down):
        row = []
        for j in range(left, right):
            row.append(image[i][j])
        ris.append(row)
    return ris


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
    image = __read_image(imagepath)
    image_height = len(image)
    image_width = len(image[0])
    crops = []
    for coord in coords:
        up, down, left, right = __get_bounds(coord)
        up -= (down-up)*(enlarge/2)
        down += (down-up)*(enlarge/2)
        left -= (right-left)*(enlarge/2)
        right += (right-left)*(enlarge/2)
        if up < 0:
            up = 0
        if left < 0:
            left = 0
        if right > image_width:
            right = image_width
        if down > image_height:
            down = image_height
        cropped_image = __copy_area(image, up, down, left, right)
        crops.append(cropped_image)
    if save:
        split = os_p.splitext(imagepath)
        for i in range(0, len(crops)):
            io.imsave(split[0]+"_crop"+str(i)+split[1], crops[i])
    return np.array(crops)


def showimages(images):
    """being images an array of images, displays them all"""
    for image in images:
        plt.figure()
        plt.imshow(image)
        plt.show()


showimages(cropimage("VOC2010_150.jpg", "VOC2010_150.mat", save=True))