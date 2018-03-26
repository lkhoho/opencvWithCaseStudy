from matplotlib import pylab as plt
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)

eq = cv2.equalizeHist(gray)
cv2.imshow('Histogram Equalization', np.hstack([gray, eq]))
cv2.waitKey(0)
