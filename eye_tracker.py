import cv2
import collections

"""
Class to track eye movements. It uses haar_cascades and meanshift.
It has two states. Detect and track.
1. We detect face present in the frame, then we detect eyes 
present within the bounding box of that face.
2. We use mean shift to keep a track of the eyes over the next few
frames.
Since mean shift is faster than cascades, this should be smoother
Also, at angles, the detection does not work so well in my
experience.
"""
class EyeTracker():
    def __init__(self, face_casc_path, eyes_casc_path, detect_frequency):
        self.face_cascade = cv2.CascadeClassifier(face_casc_path)
        self.eyes_cascade = cv2.CascadeClassifier(eyes_casc_path)
        self.detect_frequency = detect_frequency
        self.frame_counter = 0
        self.states = ['detect', 'track']
        self.current_state = self.states[0]
        self.eye_positions = None
        return

    # Function to be called every frame
    def get_position(self, frame):
        self.frame_counter += 1
        if self.frame_counter >= self.detect_frequency:
            self.frame_counter = 0
            self.current_state = self.states[0]
        if self.current_state == 'detect':
            self.detect(frame)
        elif self.current_state == 'track':
            self.track(frame)
        return self.eye_positions

    # First detect a face, then crop the frame,
    # and detect eyes within that. While returning, ensure 
    # that you return the original uncropped coordinates
    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.8,
            minNeighbors=3,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        if len(faces):
            face_x, face_y ,face_w ,face_h = faces[0]
            cv2.line(frame, (face_x, face_y), (face_x+face_w, face_y), (255,255,0))
            frame = frame[face_y:face_y+face_h, face_x:face_x+face_w]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eyes = self.eyes_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=3,
                minSize=(20, 20),
                flags=cv2.cv.CV_HAAR_SCALE_IMAGE
            )
            if len(eyes) == 2:
                self.eye_positions = []
                # self.current_state = 'track'
                for eye_x, eye_y, eye_w, eye_h in eyes:
                    self.eye_positions.append([face_x+eye_x, face_y+eye_y, eye_w, eye_h])
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # eyes = self.eyes_cascade.detectMultiScale(
        #     gray,
        #     scaleFactor=2,
        #     minNeighbors=1,
        #     minSize=(20, 20),
        #     flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        # )
        # if len(eyes) == 2:
        #     self.eye_positions = []
        #     # self.current_state = 'track'
        #     for eye_x, eye_y, eye_w, eye_h in eyes:
        #         self.eye_positions.append([eye_x, eye_y, eye_w, eye_h])
        return



