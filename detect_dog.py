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
import io
import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

#MQTT_HOST="mosquitto-service"
#MQTT_PORT=1883
#MQTT_TOPIC = "dogdetect/images"


#face_cascade = cv.CascadeClassifier('best.pt')
#model = DetectMultiBackend('best.pt')

cap = cv.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #dog = model.detectMultiScale(gray, 1.3, 5)
    pilImage = Image.fromarray(image)

    cv.imshow('frame',image)

    # Convert to JPEG Buffer
    buffered = io.BytesIO()
    pilImage.save(buffered, quality=100, format="JPEG")

    # Build multipart form and post request
    m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})

    response = requests.post("https://detect.roboflow.com/dog_labels/2?api_key=Vw7zyUhuuduWIuv9krcB", data=m, headers={'Content-Type': m.content_type})


    print(response)
    print(response.json())
        
    #for (x,y,w,h) in dog:
        #cv.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
        #dog = gray[y:y+h, x:x+w]

        #rc,png = cv.imencode('.png', dog)
        #msg = png.tobytes()
        #client.publish(MQTT_TOPIC, msg, qos=1, retain = False)
        #print('face', dog)
        #print('message sent')

    # close connection
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
