import numpy as np
import cv2

video_capture = cv2.VideoCapture(0)

# The idea is that we want to search for a face/eyes
# Once we've found that, we want to set up camshift
# to track that one.
face_coordinates = []
cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = video_capture.read()
    if not face_coordinates:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            face_coordinates =((x, y), (x+w, y+h))
            track_window = (x,y,w,h)
            roi = frame[y:y+w, x:x+w]
            hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
            roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
            cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
            break
    else:
        cv2.rectangle(frame, face_coordinates[0], face_coordinates[1], (0, 255, 0), 2)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        pts = (int(ret[0][0]), int(ret[0][1]), int(ret[1][0]), int(ret[1][1]))
        # pts = (int(ret[0][0]), int(ret[0][1]), int(ret[0][0])+int(ret[1][0]), int(ret[0][1]+int(ret[1][1])))
        # Draw it on image
        cv2.line(frame, (pts[0], pts[1]), (pts[2], pts[3]), (255,255,255), 2)
        # cv2.line(frame, (pts[0], pts[2]), (pts[1], pts[3]), (255,255,255), 2)
        # cv2.line(frame, (pts[0], pts[2]), (pts[1], pts[3]), (255,255,255), 2)
        # cv2.line(frame, (pts[0], pts[2]), (pts[1], pts[3]), (255,255,255), 2)
        # pts = cv2.boxPoints(ret)
        # pts = np.int0(pts)
        # cv2.polylines(frame,[pts],True, 255,2)

    # Display the resulting frame
    cv2.imshow('meanshift', (cv2.flip(frame, 1)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
