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
#from torch2trt import TRTModule

#model_trt = TRTModule()
#model_trt.load_state_dict(torch.load('efficientdet-d0_74_17600.pth'))
doggies = cv.CascadeClassifier('efficientdet-d0_74_17600.xml')

cap = cv.VideoCapture(0)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #dogs = model_trt.detectMultiScale(image, (1.1, 3))

    # Don't do anything if there's 
    # no dog
    #amount_found = len(dog)
	  
    #if amount_found != 0:
      
    cv.imshow('frame',image)
    #for (x,y,w,h) in dogs:
        #cv.rectangle(image,(x,y),(x+w,y+h),(416,0,0),2)
        #dog = image[y:y+h, x:x+w]
    #rc,png = cv.imencode('.png', image)
    #msg = png.tobytes()
    #client.publish(MQTT_TOPIC, msg, qos=1, retain = False)
    #print('dog', dog)
    print('message sent')
        
        
        # close connection
    if cv.waitKey(1) & 0xFF == ord('q'):
       break


