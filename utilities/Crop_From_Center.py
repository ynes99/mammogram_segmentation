def crop_image_from_center(img, center, shape):
    y = center[0]
    x = center[1]
    h = shape[0]
    w = shape[1]
    crop_img = img[int(float(y - h / 2)):int(float(y + h / 2)), int(float(x - w / 2)):int(float(x + w / 2))]
    return crop_img
