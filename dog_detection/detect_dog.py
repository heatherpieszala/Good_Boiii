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

MQTT_HOST="mosquitto-service"
MQTT_PORT=1883
MQTT_TOPIC = "dogdetect/images"


#face_cascade = cv.CascadeClassifier('best.pt')
#model = DetectMultiBackend('best.pt')

cap = cv.VideoCapture(0)

#from client connections pyton -steves internet guide
def on_log(client, userdata, level, buf):
    print("log: ", buf)

def on_disconnect(client, userdata, rc):
    logging.info("disconnecting reason  "  +str(rc))
    client.connected_flag=False
   # client.disconnect_flag=True

def on_connect(client, userdata, flags, rc):
    if rc==0: #0: Connection successful
        client.connected_flag=True #set flag
        print("connected OK Returned code=",rc)
    else:
        print("Bad connection Returned code=",rc)
        client.bad_connection_flag=True

mqtt.Client.connected_flag=False
mqtt.Client.bad_connection_flag=False


#phao-MQTT client
client = mqtt.Client()
client.on_connect = on_connect
print("connecting to broker")
client.on_disconnect = on_disconnect
client.on_log = on_log

client.loop_start()

#connect to client
try:
    client.connect(MQTT_HOST, MQTT_PORT, 60)
except:
    print("can't connect")
    sys.exit(1)

while not client.connected_flag: #wait in loop
    print("waiting in loop")
    time.sleep(1)
    if client.bad_connection_flag:
        client.loop_stop()
        sys.exit()
print("back in main loop")

time.sleep(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #dog = model.detectMultiScale(gray, 1.3, 5)
    pilImage = Image.fromarray(image)

    # Convert to JPEG Buffer
    buffered = io.BytesIO()
    pilImage.save(buffered, quality=100, format="JPEG")

    # Build multipart form and post request
    m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})

    response = requests.post("https://detect.roboflow.com/dog_labels/2?api_key=Vw7zyUhuuduWIuv9krcB", data=m, headers={'Content-Type': m.content_type})


    print(response)
    print(response.json())
        
    #for (x,y,w,h) in dog:
        #cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        #dog = gray[y:y+h, x:x+w]

    rc,png = cv.imencode('.png', image)
    msg = png.tobytes()
    client.publish(MQTT_TOPIC, msg, qos=1, retain = False)
        #print('face', dog)
        #print('message sent')

    # close connection
    #if cv.waitKey(1) & 0xFF == ord('q'):
       # break

time.sleep(1)

client.loop_stop()
client.disconnect()
cap.release()
cv.destroyAllWindows()
