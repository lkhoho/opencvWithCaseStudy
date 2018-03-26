import argparse
import numpy as np
import cv2
from utils.imutils import resize


ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', help='Path to the (optional) video file')
args = vars(ap.parse_args())

redLower = np.array([21, 14, 100], dtype='uint8')
redUpper = np.array([60, 73, 250], dtype='uint8')

if not args.get('video', False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args['video'])

while True:
    (isGrabbed, frame) = camera.read()

    if not isGrabbed:
        break

    frame = resize(frame, width=600)

    blue = cv2.inRange(frame, redLower, redUpper)
    blue = cv2.GaussianBlur(blue, (3, 3), 0)
    (_, cnts, _) = cv2.findContours(blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) > 0:
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))
        cv2.drawContours(frame, [rect], -1, (0, 255, 0), 2)

    cv2.imshow('Tracking', frame)
    cv2.imshow('Binary', blue)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
