#!/usr/bin/python
"""
Mean shift running on each eye. Press 'a' once the eyes are being detected correctly.
"""
def to_int(x):
    a = [int(i) for i in x]
    return tuple(a)

def draw_line(points, color=(255,255,255), thickness=2):
    cv2.line(frame, points[0], points[1], color, thickness)
    return

def draw_box(rect_details, color=(255,255,255), thickness=2):
    draw_line([(rect_details[0],rect_details[1]), (rect_details[0]+rect_details[2],rect_details[1])], color, thickness)
    draw_line([(rect_details[0]+rect_details[2],rect_details[1]), (rect_details[0]+rect_details[2],rect_details[1]+rect_details[3])], color, thickness)
    draw_line([(rect_details[0]+rect_details[2],rect_details[1]+rect_details[3]), (rect_details[0],rect_details[1]+rect_details[3])], color, thickness)
    draw_line([(rect_details[0],rect_details[1]+rect_details[3]), (rect_details[0],rect_details[1])], color, thickness)
    return

class Shifter():
    def __init__(self, track_window, roi_hist):
        self.track_window = track_window
        self.roi_hist = roi_hist
        self.dst = None
        return

import cv2
import sys
import numpy as np

cascPath = 'haarcascade_eyes.xml'
eyes_cascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

frame_catcher = 0
measurement_frequency = 2
points = []

capturing_eyes = True
cam_shifting = False

while capturing_eyes:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # We only want the computation done every measurement_frequency frames
    frame_catcher += 1
    if frame_catcher == measurement_frequency or not points:
        frame_catcher = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eyes_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=3,
            minSize=(20, 20),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        points = []
        windows = []
        # Mark centre of each eye
        for (x, y, w, h) in eyes:
            points.append((x + (w/2), y + (h/2)))
            windows.append((x,y,w,h))

    # Draw the line between eyes
    if len(windows) >= 2:
        draw_line(points)
        draw_box(windows[0])
        draw_box(windows[1])
    # Display the resulting frame
    cv2.imshow('Mirror', (cv2.flip(frame, 1)))
    if cv2.waitKey(1) == ord('a'):
        if len(points) >= 2:
            capturing_eyes = False
            cam_shifting = True
    elif cv2.waitKey(1) == ord('q'):
        break

eyes = []
for window in windows[:2]:
    c,r,w,h = window
    track_window = (window)
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    eyes.append(Shifter(track_window, roi_hist))
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while cam_shifting:
    ret, frame = video_capture.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for eye in eyes:
        eye.dst = cv2.calcBackProject([hsv],[0],eye.roi_hist,[0,180],1)
        ret, eye.track_window = cv2.meanShift(eye.dst, eye.track_window, term_crit)
        draw_box(eye.track_window)
    
    cv2.imshow('Mirror', (cv2.flip(frame, 1)))
    if cv2.waitKey(1) == ord('q'):
        break


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()