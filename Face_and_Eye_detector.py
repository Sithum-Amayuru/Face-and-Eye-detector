import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the classifiers using relative paths
face_clsfr = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_clsfr = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

class FaceEyeDetector(VideoTransformerBase):
    def transform(self, frame):
        # Convert the incoming video frame to a NumPy array (BGR format)
        img = frame.to_ndarray(format="bgr24")
        
        img = cv2.resize(img, (0, 0), fx=1, fy=1)
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_clsfr.detectMultiScale(imgray, 1.1, 5, minSize=(150, 150), maxSize=(350, 350))
        
        for face in faces:
            x = face[0]
            y = face[1]
            w = face[2]
            h = face[3]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, 'FACE', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            face_img = imgray[y:y + h, x:x + w]
            eyes = eye_clsfr.detectMultiScale(face_img, 1.1, 5, minSize=(10, 10), maxSize=(100, 100))
            
            for eye in eyes:
                ex = eye[0]
                ey = eye[1]
                ew = eye[2]
                eh = eye[3]
                cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 255), 2)
                cv2.putText(img, 'EYE', (x + ex, y + ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return img

# Streamlit Web UI Layout
st.title("Real-Time Face & Eye Detection App")
st.write("Click **Start** below to allow camera access and run the detector live via your link!")

# This handles the browser webcam feed and passes each frame through your code class
webrtc_streamer(key="face-eye-detector", video_transformer_factory=FaceEyeDetector)
