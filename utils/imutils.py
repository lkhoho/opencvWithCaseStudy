import numpy as np
import cv2


def translate(image, x, y):
    m = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, m, (image.shape[1], image.shape[0]))
    return shifted


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if not center:
        center = (w // 2, h // 2)

    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, m, (w, h))
    return rotated


def resize(image, width=None, height=None, interpolation=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if not width and not height:
        return image

    if not width:
        r = height / float(h)
        dim = (int(w * r), height)

    if not height:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=interpolation)
    return resized
