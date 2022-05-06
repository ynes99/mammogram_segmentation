import cv2
from Traitement_image.Preprocessing_functions.ApplyMask import ApplyMask
from Traitement_image.Preprocessing_functions.CLAHE import clahe
from Traitement_image.Preprocessing_functions.Crop_Borders import CropBorders
from Traitement_image.Preprocessing_functions.Flipping import flip
from Traitement_image.Preprocessing_functions.Global_Binarize import GlobalBinarise
from Traitement_image.Preprocessing_functions.Largest_Blob import XLargestBlobs
from Traitement_image.Preprocessing_functions.Normalise import minmaxnormalise
from Traitement_image.Preprocessing_functions.Open_Mask import openmask
from Traitement_image.Preprocessing_functions.Padding import pad


def full_preprocessing(image, gt_mask):
    crop_image = CropBorders(image)
    minmax_mask = minmaxnormalise(crop_image)
    mask_bin = GlobalBinarise(minmax_mask, 0.1, 1.0)
    mask_opened = openmask(mask=mask_bin, ksize=(33, 33), operation="open")
    mask_blob = XLargestBlobs(mask=mask_opened, top_X=1)[1]
    clean_image = ApplyMask(img=crop_image, mask=mask_blob)
    gt_mask, flipped_image = flip(clean_image, mask_blob, gt_mask)
    denoise_image = cv2.medianBlur(flipped_image, 3)
    contrast_enhanced_image = clahe(img=denoise_image)
    final_image = pad(img=contrast_enhanced_image)
    final_gt_mask = pad(img=gt_mask)
    # final_image = minmaxnormalise(padded_image)
    return final_image, final_gt_mask
