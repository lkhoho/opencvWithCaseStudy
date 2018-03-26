from matplotlib import pylab as plt
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)

channels = cv2.split(image)
colors = ('b', 'g', 'r')
plt.figure()
plt.title('"Flattened" Color Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')

for (channel, color) in zip(channels, colors):
    hist = cv2.calcHist([channel], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

plt.show()
cv2.waitKey(0)

# draw 2D histograms
binSize = 64
fig = plt.figure()

ax = fig.add_subplot(131)
hist = cv2.calcHist([channels[0], channels[1]], channels=[0, 1], mask=None, histSize=[binSize, binSize], ranges=[0, 256, 0, 256])
p = ax.imshow(hist, interpolation='nearest')
ax.set_title('2D Color Histogram for B and G')
ax.set_xlabel('# of Pixels in B')
ax.set_ylabel('# of Pixels in G')
plt.colorbar(p)

ax = fig.add_subplot(132)
hist = cv2.calcHist([channels[1], channels[2]], channels=[0, 1], mask=None, histSize=[binSize, binSize], ranges=[0, 256, 0, 256])
p = ax.imshow(hist, interpolation='nearest')
ax.set_title('2D Color Histogram for G and R')
ax.set_xlabel('# of Pixels in G')
ax.set_ylabel('# of Pixels in R')
plt.colorbar(p)

ax = fig.add_subplot(133)
hist = cv2.calcHist([channels[0], channels[2]], channels=[0, 1], mask=None, histSize=[binSize, binSize], ranges=[0, 256, 0, 256])
p = plt.imshow(hist, interpolation='nearest')
ax.set_title('2D Color Histogram for B and R')
ax.set_xlabel('# of Pixels in B')
ax.set_ylabel('# of Pixels in R')
plt.colorbar(p)

plt.tight_layout()
plt.show()
print('2D histogram shape: {}, with {} values.'.format(hist.shape, hist.flatten().shape[0]))
cv2.waitKey(0)

# draw a 3D histogram
plt.figure()
hist = cv2.calcHist([image], channels=[0, 1, 2], mask=None, histSize=[8, 8, 8], ranges=[0, 256, 0, 256, 0, 256])
# plt.scatter(hist[0], hist[1], hist[2])
plt.show()
print('3D histogram shape: {}, with {} values.'.format(hist.shape, hist.flatten().shape[0]))
cv2.waitKey(0)
