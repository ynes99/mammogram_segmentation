import cv2
import numpy as np
from matplotlib import image as mpimg

from ..utilities.Path_Join import p_join


def sum_masks(mask_list):
    """
    mask_list : {list of numpy.ndarray}
        A list of masks (numpy.ndarray) that needs to be summed.
    Returns
    -------
    summed_mask_bw: {numpy.ndarray}
        The summed mask, ranging from [0, 1].
    """
    summed_mask = np.zeros(mask_list[0].shape)

    for arr in mask_list:
        summed_mask = np.add(summed_mask, arr)

    # Binarize (there might be some overlap, resulting in pixels with
    # values of 510, 765, etc...)
    _, summed_mask_bw = cv2.threshold(src=summed_mask, thresh=1, maxval=255, type=cv2.THRESH_BINARY)
    return summed_mask_bw


def whole_mask(directory, number_masks):
    basename_file = r'\GT_mask.jpg'
    list_mask_path = []

    for i in range(number_masks):
        temp = p_join(directory, f"abnormality_{i}")
        name_file = p_join(temp, basename_file)
        print(name_file)
        list_mask_path.append(name_file)
    if len(list_mask_path) != 1:
        print(list_mask_path)
    else:
        print('There is only one mask no need to sum anything')
        return "sad"
    mask_list = [mpimg.imread(path) for path in list_mask_path]
    return sum_masks(mask_list)
