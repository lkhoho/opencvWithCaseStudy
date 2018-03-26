import cv2


class RGBHistogram(object):
    """
    Calculate 3D histogram of RGB colors of an image.
    """

    def __init__(self, bins):
        self.bins = bins

    def describe(self, image, mask=None):
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)

        return hist.flatten()
