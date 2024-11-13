# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 08:52:03 2024

@author: Lenovo
"""

import numpy as np
import cv2
import sys, inspect, os
import argparse

cmd_subfolder = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..", "..", "Image_Lib")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import image_utils as utils
import EyeTrackingLib as tracker

ap = argparse.ArgumentParser("Track eyes in video input")
ap.add_argument("-v", "--video", help="Path to video file. Defaults to webcam video")

args = vars(ap.parse_args())

if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

def add_text(image, text):
    cv2.putText(image, text, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def image_resize(image, width=-1, height=-1):
    shape = image.shape
    if width == -1:
        if height == -1:
            return image
        else:
            return cv2.resize(image, (int(height * shape[1] / shape[0]), height))
    elif height == -1:
        return cv2.resize(image, (width, int(width * shape[0] / shape[1])))
    else:
        cv2.resize(image, (width, height))

face_cascade = cv2.CascadeClassifier('C:/Users/Lenovo/Downloads/haarcascade_frontalface_default.xml')
tracker.showImage = False

while True:
    _, frame = camera.read()
    """if not grabbed:
        print ("Camera read failed!")
        break"""
    #frame = camera.read()
    

    #frame = image_resize(frame, width=600, height=600)
    grayA = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(grayA)
    face_box = None
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    if len(faces) > 0:
        face_box = max(faces, key=lambda item: item[2] * item[3])

    if face_box is not None:
        x, y, w, h = face_box
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
        roi_x = int(x + w * 0.14)
        roi_y = int(y + h * 0.25)
        roi_w = int(w * 0.3)
        roi_h = int(h * 0.27)
        
         

        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
        row, col = tracker.find_eye_center(gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w])
        # print row, col
        roi_xx = int(roi_x + col)
        roi_yy = int(roi_y + row)

        cv2.circle(frame, (roi_xx, roi_yy), 5, (0, 255, 0), -1)

        roi_x = x + w - roi_w - int(w * 0.13)
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 0), 2)
        row, col = tracker.find_eye_center(gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w])
        # print row, col
        
        roi_xx = int(roi_x + col)
        

        cv2.circle(frame, (roi_xx, roi_yy), 5, (0, 0, 255), -1)

    else:
        add_text(frame, "Face not found!")
        # row, col = tracker.find_eye_center(gray[:, 0:frame.shape[1] / 2])
        # print row, col
        #
        # cv2.circle(frame, (col, row), 10, (255, 255, 0), -1)
        #
        # row, col = tracker.find_eye_center(gray[:, frame.shape[1] / 2:])
        # print row, col
        #
        # cv2.circle(frame, (frame.shape[1] / 2 + col, row), 10, (255, 0, 0), -1)

    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()