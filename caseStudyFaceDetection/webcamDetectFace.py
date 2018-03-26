from utils.facedetector import FaceDetector
from utils.imutils import resize
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--face', required=True, help='Path to where the face cascade resides')
ap.add_argument('-v', '--video', help='Path to the (optional) video file')
args = vars(ap.parse_args())

fd = FaceDetector(faceCascadePath=args['face'])

if not args.get('video', False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args['video'])

while True:
    (isGrabbed, frame) = camera.read()

    if args.get('video') and not isGrabbed:
        break

    frame = resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceRects = fd.detect(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    frameCopy = frame.copy()

    for (fX, fY, fW, fH) in faceRects:
        cv2.rectangle(frameCopy, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

    cv2.imshow('Face', frameCopy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
