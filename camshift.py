"""
Trying to capture the face and then use camshift to get the change in angle.
"""

def draw_line(pt1, pt2):
    cv2.line(frame, pt1, pt2, (255,255,255), 2)
    return

def sin(x):
    return math.sin(math.radians(x))
def cos(x):
    return math.cos(math.radians(x))

def draw_rotated_rect(ret):
    center = ret[0]
    sides = ret[1]
    angle = ret[2]
    pt1 = (int(center[0] + .5*sides[0]*sin(angle)), int(center[1] + .5*sides[1]*sin(angle)))
    pt2 = (int(center[0] - .5*sides[0]*sin(angle)), int(center[1] + .5*sides[1]*sin(angle)))
    pt3 = (int(center[0] - .5*sides[0]*sin(angle)), int(center[1] - .5*sides[1]*sin(angle)))
    pt4 = (int(center[0] + .5*sides[0]*sin(angle)), int(center[1] - .5*sides[1]*sin(angle)))
    draw_line(pt1, pt2)
    draw_line(pt2, pt3)
    draw_line(pt3, pt4)
    draw_line(pt4, pt1)
    return

def draw_median_line(ret):
    center = (int(ret[0][0]), int(ret[0][1]))
    sides = ret[1][1]/2
    angle = ret[2]
    top = (int(center[0]+sides*sin(x)), int(center[1]+sides*cos(x)))
    draw_line(center, top)
    return



import cv2
import sys
import math
import numpy as np

cascPath = 'haarcascade_frontalface_default.xml'
eyes_cascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

frame_catcher = 0
measurement_frequency = 2
face_detected = False

capturing_face = True
cam_shifting = False

while capturing_face:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # We only want the computation done every measurement_frequency frames
    frame_catcher += 1
    if frame_catcher == measurement_frequency or not face_detected:
        frame_catcher = 0
        face_detected = False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = eyes_cascade.detectMultiScale(
            gray,
            scaleFactor=1.8,
            minNeighbors=3,
            minSize=(50, 50),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        if face_detected is not False:
            face_detected = True

    # Draw a rectangle around the faces
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow('Mirror', (cv2.flip(frame, 1)))
    if cv2.waitKey(1) == ord('a'):
        if len(face) >= 1:
            capturing_face = False
            cam_shifting = True
    elif cv2.waitKey(1) == ord('q'):
        break

c,r,w,h = face[0]
track_window = (c,r,w,h)
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while cam_shifting:
    ret ,frame = video_capture.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    # pts = cv2.boxPoints(ret)
    # pts = np.int0(pts)
    # cv2.polylines(frame,[pts],True, 255,2)
    draw_rotated_rect(ret)
    draw_median_line(ret)
    cv2.imshow('Mirror', (cv2.flip(frame, 1)))

    if cv2.waitKey(1) == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()