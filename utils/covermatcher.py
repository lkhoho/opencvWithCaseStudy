import numpy as np
import cv2


class CoverMatcher(object):
    def __init__(self, descriptor, coverPaths, ratio=0.7, minMatches=40, useHamming=True):
        """
        Construct a CoverMatcher instance.
        :param descriptor: an instance of CoverDescriptor.
        :param coverPaths: path to the directory where the covers images are stored.
        :param ratio: the ratio of nearest neighbor distances suggested by Lowe to prune down the number of keypoints
        a homography needs to computed for.
        :param minMatches: the minimum number of matches required for a homography to be calculated.
        :param useHamming: a boolean indicating whether the Hamming or Euclidean distance should be used to compare
        feature vectors.
        """

        self.descriptor = descriptor
        self.coverPaths = coverPaths
        self.ratio = ratio
        self.minMatches = minMatches
        self.distanceMethod = 'BruteForce'

        if useHamming:
            self.distanceMethod += '-Hamming'

    def search(self, queryKps, queryDescs):
        results = {}

        for coverPath in self.coverPaths:
            cover = cv2.imread(coverPath)
            gray = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
            (kps, descs) = self.descriptor.describe(gray)
            score = self.match(queryKps, queryDescs, kps, descs)
            results[coverPath] = score

        if len(results) > 0:
            results = sorted([(v, k) for (k, v) in results.items() if v > 0], reverse=True)

        return results

    def match(self, kpsA, featuresA, kpsB, featuresB):
        """
        Compute score of matchness of two images.
        :param kpsA: the list of keypoints associated with the first image to be matched.
        :param featuresA: the list of feature vectors associated with the first image to be matched.
        :param kpsB: the list of keypoints associated with the second image to be matched.
        :param featuresB: the list of feature vectors associated with the second image to be matched.
        :return: a score representing matchness of two images. Higher score mean better match.
        """

        matcher = cv2.DescriptorMatcher_create(self.distanceMethod)
        rawMatches = matcher.knnMatch(featuresB, featuresA, k=2)  # want up to 2 nearest neighbors for each feature
        matches = []

        for m in rawMatches:
            # if there are indeed two matches, perform David Lowe's ratio test to remove false matches and prune down
            # the number of keypoints the homograpy needs to be computed for, thus speeding up the entire process
            if len(m) == 2 and m[0].distance < m[1].distance * self.ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # a check to ensure the number of matches is at least the number of minimum matches.
        if len(matches) > self.minMatches:
            # sore the (x, y) coordinates for each set of matched keypoints in ptsA and ptsB
            ptsA = np.float32([kpsA[i] for (i, _) in matches])
            ptsB = np.float32([kpsB[j] for (_, j) in matches])

            # use RANSAC (RANdom SAmple Consensus) method to compute the homography. Alternative method to use is
            # cv2.LMEDS (Least-MEDian) robust method. ransacReprojThreshold parameter represents the error tolerance.
            (_, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold=4.0)
            return float(status.sum()) / status.size

        return -1.0
