import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Image', image)
blurred = cv2.GaussianBlur(image, (5, 5), 0)

thresholdValue = 100
(t, thresh) = cv2.threshold(blurred, thresholdValue, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold Binary ({})'.format(thresholdValue), thresh)

(tInv, threshInv) = cv2.threshold(blurred, thresholdValue, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('Threshold Binary Inverse ({})'.format(thresholdValue), threshInv)

cv2.imshow('Hairs', cv2.bitwise_and(image, image, mask=threshInv))
cv2.waitKey(0)

