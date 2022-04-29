from preprocessing_functions.Normalise import MinMaxNormalise
from utilities.Crop_From_Center import crop_image_from_center
import cv2
from preprocessing_functions.CLAHE import clahe


def preproc_and_crop(image, center, shape):
    crop = crop_image_from_center(image, center=center, shape=shape)
    crop = cv2.medianBlur(crop, 3)
    crop = clahe(img=crop)
    crop = MinMaxNormalise(crop)
    return crop
