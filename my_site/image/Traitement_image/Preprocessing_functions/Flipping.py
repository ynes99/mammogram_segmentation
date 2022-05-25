from Traitement_image.Preprocessing_functions.Horizontal_flip import HorizontalFlip
import numpy as np


def flip(image, mask_blob, gt_mask):
    horizontal_flip = HorizontalFlip(mask=mask_blob)
    flipped_image = image
    if horizontal_flip:
        flipped_image = np.fliplr(image)
        gt_mask = np.fliplr(gt_mask)
    return gt_mask, flipped_image
