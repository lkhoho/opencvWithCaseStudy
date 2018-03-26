import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Image', image)
blurred = cv2.GaussianBlur(image, (5, 5), 0)

blockSize = 11
C = 4  # value subtracted from the mean
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, C)
cv2.imshow('Mean_C Thresh (BlockSize={}, C={})'.format(blockSize, C), thresh)

blockSize = 15
C = 3
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C)
cv2.imshow('Gaussian_C Thresh (BlockSize={}, C={})'.format(blockSize, C), thresh)
cv2.waitKey(0)
