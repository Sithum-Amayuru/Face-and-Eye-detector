import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import urllib.request
import os

# Automatically download Haar cascade files if they don't exist
face_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
eye_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml"

if not os.path.exists("haarcascade_frontalface_default.xml"):
    urllib.request.urlretrieve(face_url, "haarcascade_frontalface_default.xml")

if not os.path.exists("haarcascade_eye_tree_eyeglasses.xml"):
    urllib.request.urlretrieve(eye_url, "haarcascade_eye_tree_eyeglasses.xml")

face_clsfr = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_clsfr = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

class FaceEyeDetector(VideoTransformerBase):
    def __init__(self):
        self.latest_frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (0, 0), fx=1, fy=1)
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_clsfr.detectMultiScale(imgray, 1.1, 5, minSize=(150, 150), maxSize=(350, 350))
        for face in faces:
            x, y, w, h = face[0], face[1], face[2], face[3]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, 'FACE', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            face_img = imgray[y:y + h, x:x + w]
            eyes = eye_clsfr.detectMultiScale(face_img, 1.1, 5, minSize=(10, 10), maxSize=(100, 100))
            for eye in eyes:
                ex, ey, ew, eh = eye[0], eye[1], eye[2], eye[3]
                cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 255), 2)
                cv2.putText(img, 'EYE', (x + ex, y + ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Store the processed frame so we can capture it
        self.latest_frame = img
        return img

st.title("Real-Time Face & Eye Detection App")
st.write("Click **Start** below to run the detector live!")

webrtc_streamer(key="face-eye-detector", video_transformer_factory=FaceEyeDetector)
