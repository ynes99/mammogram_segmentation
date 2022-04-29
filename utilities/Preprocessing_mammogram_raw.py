import cv2

from preprocessing_functions.ApplyMask import ApplyMask
from preprocessing_functions.CLAHE import clahe
from preprocessing_functions.Crop_Borders import CropBorders
from preprocessing_functions.Flipping import flip
from preprocessing_functions.Global_Binarize import GlobalBinarise
from preprocessing_functions.Largest_Blob import XLargestBlobs
from preprocessing_functions.Normalise import MinMaxNormalise
from preprocessing_functions.Open_Mask import OpenMask
from preprocessing_functions.Padding import Pad


def full_preprocessing(image, gt_mask):
    crop_image = CropBorders(image)
    minmax_mask = MinMaxNormalise(crop_image)
    mask_bin = GlobalBinarise(minmax_mask, 0.1, 1.0)
    mask_opened = OpenMask(mask=mask_bin, ksize=(33, 33), operation="open")
    mask_blob = XLargestBlobs(mask=mask_opened, top_X=1)[1]
    clean_image = ApplyMask(img=crop_image, mask=mask_blob)
    gt_mask, flipped_image = flip(clean_image, mask_blob, gt_mask)
    denoise_image = cv2.medianBlur(flipped_image, 3)
    contrast_enhanced_image = clahe(img=denoise_image)
    padded_image = Pad(img=contrast_enhanced_image)
    final_gt_mask = Pad(img=gt_mask)
    final_image = MinMaxNormalise(padded_image)
    return final_image, final_gt_mask
