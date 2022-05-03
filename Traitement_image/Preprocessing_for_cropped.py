from Traitement_image.Crop_From_Center import crop_image_from_center
import cv2
from Traitement_image.Preprocessing_functions.CLAHE import clahe


def preproc_and_crop(image, center, shape):
    crop = crop_image_from_center(image, center=center, shape=shape)
    crop = cv2.medianBlur(crop, 3)
    crop = clahe(img=crop)
    return crop
