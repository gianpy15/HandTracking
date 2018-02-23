import source.hands_bounding_utils.utils as u
import os


def __list_objects_in_path(path):
    """lists all the objects at the given path"""
    return os.listdir(path)


def __clean_list(lista):
    """removes '__init__.py' from the given list and returns the clean list"""
    lista.remove('__init__.py')
    return lista


def get_samples_from_dataset(imagesfolderpath, annotationsfolderpath, number, height_shrink_rate=10,
                             width_shrink_rate=10, overlapping_penalty=0.9):
    """gets a number of samples from the dataset that will be used to train the network that locates hands from images,
    as first step of the computation of the 3D model of the hands
    :type imagesfolderpath: the path of THE FOLDER that contains the images.
    :type annotationsfolderpath: the path of the FOLDER that contains annotations.
    :type number: the number of images to get. If this is <=0 or >=Total_number_of_images, all images will be
    returned.
    :type height_shrink_rate: the factor that represents how much the height of heatmaps will be shrunk w.r.t.
    the original image
    :type width_shrink_rate: the factor that represents how much the width of heatmaps will be shrunk w.r.t.
    the original image
    :type overlapping_penalty: how much overlapping of hands is penalized in the heatmaps
    Note that for this function to work properly, annotations and images must eb in DIFFERENT folders and must have
    THE SAME NAME, except for the extension"""
    images_paths = __clean_list(__list_objects_in_path(imagesfolderpath))
    annots_paths = __clean_list(__list_objects_in_path(annotationsfolderpath))
    num = len(images_paths)
    if 0 < number < num:
        num = number
    images = []
    heatmaps = []
    for i in range(num):
        im = u.read_image(os.path.join(imagesfolderpath, images_paths[i]))
        images.append(im)
        heat = u.get_heatmap_from_mat(im, os.path.join(annotationsfolderpath, annots_paths[i]),
                                      height_shrink_rate, width_shrink_rate, overlapping_penalty)
        heatmaps.append(heat)
    return images, heatmaps


if __name__ == '__main__':
    im_f = "dataset/images"
    an_f = "dataset/annotations"
    print(__clean_list(__list_objects_in_path(im_f)))
    print(__clean_list(__list_objects_in_path(an_f)))
    images1, heatmaps1 = get_samples_from_dataset(im_f, an_f, -1)
    u.showimages(images1)
    for heat1 in heatmaps1:
        u.showimage(u.heatmap_to_rgb(heat1))
