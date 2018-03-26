import argparse
import utils.imutils as imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)
cv2.waitKey(0)

shifted = imutils.translate(image, 25, 50)
cv2.imshow('Shifted Down and Right', shifted)
cv2.waitKey(0)

shifted = imutils.translate(image, -50, -90)
cv2.imshow('Shifted Up and Left', shifted)
cv2.waitKey(0)

shifted = imutils.translate(image, 0, 100)
cv2.imshow('Shifted Down', shifted)
cv2.waitKey(0)
