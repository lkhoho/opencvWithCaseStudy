import matplotlib.pylab as plt
import numpy as np
import argparse
import cv2


def plotHistogram(image, title, mask=None):
    channels = cv2.split(image)
    colors = ('b', 'g', 'r')
    plt.figure()
    plt.title(title)
    plt.xlabel('Bins')
    plt.ylabel('# of Pixels')

    for (channel, color) in zip(channels, colors):
        hist = cv2.calcHist([channel], channels=[0], mask=mask, histSize=[256], ranges=[0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    plt.show()


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)
plotHistogram(image, 'Histogram for Original Image')
cv2.waitKey(0)

mask = np.zeros(image.shape[:2], dtype='uint8')
cv2.rectangle(mask, (384, 53), (621, 354), 255, -1)
cv2.imshow('Mask', mask)

masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow('Applying the Mask', masked)
plotHistogram(image, 'Histogram for Masked Image', mask=mask)
cv2.waitKey(0)
