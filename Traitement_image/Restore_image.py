from PIL import Image, ImageOps
from matplotlib import cm
import numpy as np


def restore_mask(mask_path_gt_path, result_segmentation_path, shape_roi, center_roi):

    # Opening the primary image (used in background)
    image = Image.open(mask_path_gt_path)
    width, height = image.size
    myarray = np.zeros((height, width))
    img1 = Image.fromarray(np.uint8(cm.gist_earth(myarray) * 255))

    # Opening the secondary image (overlay image)
    img2 = Image.open(result_segmentation_path)
    img2 = ImageOps.grayscale(img2)

    # Pasting img2 image on top of img1
    # starting at coordinates (0, 0)
    y = center_roi[1]
    x = center_roi[0]
    h = shape_roi[0]
    w = shape_roi[1]
    img1.paste(img2, (y - (h // 2), x - (w // 2)), mask=img2)
    return img1
