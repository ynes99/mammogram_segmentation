def ApplyMask(img, mask):
    """
    This function applies a mask to a given image. White
    areas of the mask are kept, while black areas are
    removed.
    """

    masked_img = img.copy()
    masked_img[mask == 0] = 0

    return masked_img