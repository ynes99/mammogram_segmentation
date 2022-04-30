import numpy as np


def pad(img):
    """
    This function pads a given image with black pixels,
    along its shorter side, into a square and returns
    the square image.
    """

    nrows, ncols = img.shape

    # If padding is required...
    if nrows != ncols:

        # Take the longer side as the target shape.
        if ncols < nrows:
            target_shape = (nrows, nrows)
        elif nrows < ncols:
            target_shape = (ncols, ncols)

        # Pad.
        padded_img = np.zeros(shape=target_shape)
        padded_img[:nrows, :ncols] = img

        return padded_img

    # If padding is not required, return original image.
    elif nrows == ncols:

        return img
