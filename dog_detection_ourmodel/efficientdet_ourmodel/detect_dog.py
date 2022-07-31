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

import torch
from torch.backends import cudnn

from backbone import EfficientDetBackbone
import matplotlib.pyplot as plt
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

MQTT_HOST="mosquitto-service"
MQTT_PORT=1883
MQTT_TOPIC = "dogdetect/images"


#direct efficientdet
compound_coef = 0
force_input_size = None  # set None to use default size

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = [ 'other', 'sit' ]

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
    img_path = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # tf bilinear interpolation is different from any other's, just make do
    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
    ori_imgs, framed_imgs, framed_metas = preprocess_video(img_path, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),

                             # replace this part with your project's anchor config
                             ratios=[(1.0, 1.0), (1.3, 0.8), (1.9, 0.5)],
                             scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    model.load_state_dict(torch.load('efficientdet-d0_74_17600.pth'))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    with torch.no_grad():
        features, regression, classification, anchors = model(x)
    
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                      anchors, regression, classification,
                      regressBoxes, clipBoxes,
                      threshold, iou_threshold)

    out = invert_affine(framed_metas, out)

    for i in range(len(ori_imgs)):
        if len(out[i]['rois']) == 0:
            continue
        ori_imgs[i] = ori_imgs[i].copy()
        for j in range(len(out[i]['rois'])):
            (x1, y1, x2, y2) = out[i]['rois'][j].astype(np.int)
            cv.rectangle(ori_imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[out[i]['class_ids'][j]]
            score = float(out[i]['scores'][j])
            print(obj)
            print('SCORE',score)

            cv.putText(ori_imgs[i], '{}, {:.3f}'.format(obj, score),
                    (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1)

            plt.imshow(ori_imgs[i])
      
    cv.imshow('frame',image)
    #for (x,y,w,h) in dogs:
        #cv.rectangle(image,(x,y),(x+w,y+h),(416,0,0),2)
        #dog = image[y:y+h, x:x+w]
    rc,png = cv.imencode('.png', image)
    msg = png.tobytes()
    client.publish(MQTT_TOPIC, msg, qos=1, retain = False)
    #print('dog', dog)
    print('message sent')
        
        
        # close connection
    if cv.waitKey(1) & 0xFF == ord('q'):
       break

time.sleep(1)

client.loop_stop()
client.disconnect()
cap.release()
cv.destroyAllWindows()
