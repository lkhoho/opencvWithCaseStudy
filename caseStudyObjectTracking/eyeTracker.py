import argparse
import numpy as np
import cv2
from utils.imutils import resize
from utils.eyetracker import EyeTracker


ap = argparse.ArgumentParser()
ap.add_argument('-f', '--face', required=True, help='Path to where the face cascade resides')
ap.add_argument('-e', '--eye', required=True, help='Path to where the eye cascade resides')
ap.add_argument('-v', '--video', help='Path to the (optional) video file')
args = vars(ap.parse_args())

et = EyeTracker(args['face'], args['eye'])

if not args.get('video', False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args['video'])

while True:
    (isGrabbed, frame) = camera.read()

    if args.get('video') and not isGrabbed:
        break

    frame = resize(frame, 600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = et.track(gray)

    for rect in rects:
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

    cv2.imshow('Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
