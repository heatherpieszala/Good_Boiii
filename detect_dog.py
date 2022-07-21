"""
The purpose of this script is to detect dogs and sit or other behavior
from a USB camera attached to a Jetson Nano in a constant stream.
We then want to publish the faces to the MQTT_BROKER.

MQTT_HOST EXAMPLES:
MQTT_HOST = "localhost"
MQTT_HOST = kubernetes service name
MQTT_HOST = an ip address
"""

import numpy as np
import cv2 as cv
import paho.mqtt.client as mqtt
import time
import sys

#MQTT_HOST="mosquitto-service"
#MQTT_PORT=1883
#MQTT_TOPIC = "dogdetect/images"


#face_cascade = cv.CascadeClassifier('best.pt')
model = DetectMultiBackend('best.pt')

cap = cv.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    dogs = model.detectMultiScale(gray, 1.3, 5)

    #cv.imshow('frame',gray)
    for (x,y,w,h) in dog:
        cv.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
        dog = gray[y:y+h, x:x+w]

        rc,png = cv.imencode('.png', dog)
        #msg = png.tobytes()
        cv.imshow('frame',gray)
        #client.publish(MQTT_TOPIC, msg, qos=1, retain = False)
        print('face', dog)
        #print('message sent')

    # close connection
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
