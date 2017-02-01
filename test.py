from eye_tracker import EyeTracker
import cv2

def get_centres(pos):
    return [(pos[0][0] + pos[0][2]/2, pos[0][1] + pos[0][3]/2), (pos[1][0] + pos[1][2]/2, pos[1][1] + pos[1][3]/2)]
video_capture = cv2.VideoCapture(0)
tracker = EyeTracker('haarcascade_frontalface_default.xml', 'haarcascade_eyes.xml', 0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    pos = tracker.get_position(frame)
    if pos:
        points = get_centres(pos)
        cv2.line(frame, points[0], points[1], (255, 255, 255), 2)
    cv2.imshow('Mirror', (cv2.flip(frame, 1)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

