def minmaxnormalise(img):
    """
    Cette fonction applique la min-max normalisation sur l'image donnée.
    """
    norm_img = (img - img.min()) / (img.max() - img.min())
    return norm_img