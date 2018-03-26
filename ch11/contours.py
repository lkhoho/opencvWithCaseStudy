import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to the image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Image', gray)
blurred = cv2.GaussianBlur(gray, (9, 9), 0)

edged = cv2.Canny(blurred, 30, 125)
cv2.imshow('Edges', edged)

(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print('I count {} contours in this image'.format(len(cnts)))

contouredImage = image.copy()
cv2.drawContours(contouredImage, cnts, -1, (0, 255, 0), 2)
cv2.imshow('Contoured', contouredImage)
cv2.waitKey(0)

for (i, c) in enumerate(cnts[:5]):
    (x, y, w, h) = cv2.boundingRect(c)
    print('Contour #{}'.format(i + 1))
    piece = image[y:y + h, x:x + w]
    cv2.imshow('Contour', piece)

    mask = np.zeros(image.shape[:2], dtype='uint8')
    ((centerX, centerY), radius) = cv2.minEnclosingCircle(c)
    cv2.circle(mask, (int(centerX), int(centerY)), int(radius), 255, -1)
    mask = mask[y:y + h, x:x + w]
    cv2.imshow('Masked Contour', cv2.bitwise_and(piece, piece, mask=mask))
    cv2.waitKey(0)
