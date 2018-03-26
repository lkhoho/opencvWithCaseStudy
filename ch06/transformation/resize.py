import argparse
import utils.imutils as imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)
cv2.waitKey(0)

newWidth = 150
r = float(newWidth) / image.shape[1]
dim = (newWidth, int(image.shape[0] * r))

resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('Resized (Width)', resized)
cv2.waitKey(0)

newHeight = 50
r = float(newHeight) / image.shape[0]
dim = (int(image.shape[1] * r), newHeight)

resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('Resized (Height)', resized)
cv2.waitKey(0)

resized = imutils.resize(image, width=100)
cv2.imshow('Resized via Function', resized)
cv2.waitKey(0)
