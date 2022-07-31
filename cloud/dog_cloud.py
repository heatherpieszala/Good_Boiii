import paho.mqtt.client as mqtt
import numpy as np 
import cv2 as cv
import time
import sys
import logging
import boto3
from botocore.exceptions import ClientError
import io

MQTT_HOST="mosquitto-service"
MQTT_PORT=1883
LOCAL_MQTT_TOPIC="dogcapture/images"

def on_log(client, userdata, level, buf):
    print("log: ", buf)

def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker with rc: " + str(rc))
        client.subscribe(LOCAL_MQTT_TOPIC)

#we want to organize and name our images coming in
image_count = 0

def on_message(client,userdata, msg):
    try:
        global image_count
        print('Got the dog!')

        if(image_count < 3):
            image_name ="dog-" + str(image_count) + ".jpg"
            file_name = "/dogs/" +image_name
            print(image_name)
            print(file_name)
        else:
            image_name = "dog-" + str(image_count) + ".jpg"
            file_name = "/dogs/" + image_name
            print(image_name)
            print(file_name)

        #upload image to s3 bucket
        s3_client = boto3.client('s3', aws_access_key_id='XXX',aws_secret_access_key='XXX')
        s3_client.upload_fileobj(io.BytesIO(msg.payload), 'dog-training-good-boiii', image_name)
        print('file upload success')
        image_count +=1
    except:
        print("Unexpected error:", sys.exc_info()[0])

local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(MQTT_HOST, MQTT_PORT, 60)
local_mqttclient.on_message = on_message
local_mqttclient.on_log = on_log



# go into a loop
local_mqttclient.loop_forever()
