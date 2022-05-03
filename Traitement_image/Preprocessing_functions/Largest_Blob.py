import cv2
import numpy as np
from Traitement_image.Preprocessing_functions.Sort_Contour_By_Area import sort_contours_by_area

def XLargestBlobs(mask, top_X=None):
    """
    This function finds contours in the given image and
    keeps only the top X largest ones.
    """

    # Trouver tout les contours de l'image binariser
    contours, hierarchy = cv2.findContours(image=mask,
                                           mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_NONE)

    n_contours = len(contours)

    # on prend seulement le plus grand contour s'il existe.
    if n_contours > 0:

        # le nombre de contours retournés est au plus égal au nombre de contours total
        if n_contours < top_X or top_X == None:
            top_X = n_contours

        # trier les contours on se basant sur la zone du contour(area contour).
        sorted_contours, bounding_boxes = sort_contours_by_area(contours=contours,
                                                                reverse=True)

        # Gon prend le contour le plus large
        X_largest_contours = sorted_contours[0:top_X]

        # Creation d'un canvas noir.
        to_draw_on = np.zeros(mask.shape, np.uint8)

        # Dessiner les plus grands contours sur le canvas.
        X_largest_blobs = cv2.drawContours(image=to_draw_on,  # Draw the contours on `to_draw_on`.
                                           contours=X_largest_contours,  # List of contours to draw.
                                           contourIdx=-1,  # Draw all contours in `contours`.
                                           color=1,  # Draw the contours in white.
                                           thickness=-1)  # Thickness of the contour lines.

    return n_contours, X_largest_blobs