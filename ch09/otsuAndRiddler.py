import argparse
import cv2
import mahotas as mh

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Image', image)
blurred = cv2.GaussianBlur(image, (5, 5), 0)

t = mh.thresholding.otsu(blurred)
print('Otsu\'s thresholding: t={}'.format(t))
thresh = image.copy()
thresh[thresh > t] = 255
thresh[thresh < 255] = 0
# thresh = cv2.bitwise_not(thresh)
cv2.imshow('Otsu', thresh)

t = mh.thresholding.rc(blurred)
print('Riddler-Calvard: t={}'.format(t))
thresh = image.copy()
thresh[thresh > t] = 255
thresh[thresh < 255] = 0
# thresh = cv2.bitwise_not(thresh)
cv2.imshow('Riddler-Calvard', thresh)
cv2.waitKey(0)

