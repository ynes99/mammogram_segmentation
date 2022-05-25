import numpy as np


def dice_coef(img, img2):
    if img.shape != img2.shape:
        raise ValueError("Shape mismatch: img and img2 must have to be of the same shape.")
    else:

        lenintersection = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if np.array_equal(img[i][j], img2[i][j]):
                    lenintersection += 1

        lenimg = img.shape[0] * img.shape[1]
        lenimg2 = img2.shape[0] * img2.shape[1]
        value = (2. * lenintersection / (lenimg + lenimg2))
        print('Dice coefficient is % s' % value)
    return value
