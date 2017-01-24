import cv2
import sys

cascPath = 'haarcascade_eyes.xml'
eyes_cascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
frame_catcher = 0
measurement_frequency = 2
points = []

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # We only want the computation done every measurement_frequency frames
    frame_catcher += 1
    if frame_catcher == measurement_frequency or not points:
        frame_catcher = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eyes_cascade.detectMultiScale(
            gray,
            scaleFactor=2,
            minNeighbors=1,
            minSize=(20, 20),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        points = []
        # Mark centre of each eye
        for (x, y, w, h) in eyes:
            points.append((x + (w/2), y + (h/2)))

    # Draw the line between eyes
    if len(points) >= 2:
        cv2.line(frame, points[0], points[1], (255, 255, 255), 2)
    # Display the resulting frame
    cv2.imshow('Mirror', (cv2.flip(frame, 1)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()