import cv2
def sort_contours_by_area(contours, reverse=True):
    """
    This function takes in list of contours, sorts them based
    on contour area, computes the bounding rectangle for each
    contour, and outputs the sorted contours and their
    corresponding bounding rectangles.
    """
    # trier les contours on se basant sur la zone du contour(area contour).
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Construire la liste des bounding boxes correspondant.
    bounding_boxes = [cv2.boundingRect(c) for c in sorted_contours]

    return sorted_contours, bounding_boxes