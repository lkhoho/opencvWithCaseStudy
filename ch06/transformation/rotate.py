import argparse
import utils.imutils as imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)
cv2.waitKey(0)

(h, w) = image.shape[:2]
center = (w // 2, h // 2)

angle = 45
rotated = imutils.rotate(image, angle)
cv2.imshow('Rotated by %s Degrees' % angle, rotated)
cv2.waitKey(0)

angle = -90
rotated = imutils.rotate(image, angle)
cv2.imshow('Rotated by %s Degrees' % angle, rotated)
cv2.waitKey(0)

angle = 180
rotated = imutils.rotate(image, angle)
cv2.imshow('Rotated by %s Degrees' % angle, rotated)
cv2.waitKey(0)
