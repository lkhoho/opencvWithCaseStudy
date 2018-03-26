import numpy as np
import cv2


class CoverDescriptor(object):
    """
    Use keypoint detectors and local invariant descriptors such as BRISK, ORB, KAZE, and AKAZE
    to describe image feature. Use of SIFT and SURF is optional and requires explicitly setting.
    """

    def __init__(self, useSIFT=False):
        self.useSIFT = useSIFT

    def describe(self, image):
        descriptor = cv2.BRISK_create()

        if self.useSIFT:
            descriptor = cv2.xfeatures2d.SIFT_create()

        (kps, descs) = descriptor.detectAndCompute(image, None)  # keypoints and descriptors
        kps = np.float32([kp.pt for kp in kps])

        return kps, descs

