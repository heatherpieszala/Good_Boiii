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
import pandas as pd

#could not get torch2trt to instlal on my jetson - aneeded tensorrt and then the whl file wasn't working to resolve error
#from torch2trt import TRTModule

#model_trt = TRTModule()
#model_trt.load_state_dict(torch.load('efficientdet-d0_74_17600.pth'))

#direct xml not working with cascade seems like we need to make cascade specific xml
#doggies = cv.CascadeClassifier('efficientdet-d0_74_17600.xml')

#primary mode to try - from this link:https://docs.opencv.org/4.x/dc/d70/pytorch_cls_tutorial_dnn_conversion.html
opencv_net = cv.dnn.readNetFromONNX('best.onnx')

#secondary - found in this post when trying to figure out solution
#opencv_net = cv.dnn_ClassificationModel('best.onnx') come back to this if readNetFromONNX not working with forward and blob

cap = cv.VideoCapture(0)

#https://github.com/opencv/opencv/blob/4.x/samples/dnn/dnn_model_runner/dnn_conversion/pytorch/classification/py_to_py_resnet50.py - from within repo in the readNetFromONNX link
def get_preprocessed_img(img):
    # read the image
    #input_img = cv.imread(img_path, cv.IMREAD_COLOR)
    input_img = img.astype(np.float32)

    input_img = cv.resize(input_img, (416, 416))

    # define preprocess parameters
    mean = np.array([0.485, 0.456, 0.406]) * 255.0
    scale = 1 / 255.0
    std = [0.229, 0.224, 0.225]

    # prepare input blob to fit the model input:
    # 1. subtract mean
    # 2. scale to set pixel values from 0 to 1
    input_blob = cv.dnn.blobFromImage(
        image=input_img,
        scalefactor=scale,
        size=(416, 416),  # img target size
        mean=mean,
        swapRB=True,  # BGR -> RGB
        crop=True  # center crop
    )
    # 3. divide by std
    input_blob[0] /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    return input_blob


def get_imagenet_labels(labels_path):
    with open(labels_path) as f:
        imagenet_labels = [line.strip() for line in f.readlines()]
    return imagenet_labels


def get_opencv_dnn_prediction(opencv_net, preproc_img, imagenet_labels):
    # set OpenCV DNN input
    opencv_net.setInput(preproc_img)

    # OpenCV DNN inference
    out = opencv_net.forward()
    print("OpenCV DNN prediction: \n")
    print("* shape: ", out.shape)

    # get the predicted class ID
    imagenet_class_id = np.argmax(out)

    # get confidence
    confidence = out[0][imagenet_class_id]
    print("* class ID: {}, label: {}".format(imagenet_class_id, imagenet_labels[imagenet_class_id]))
    print("* confidence: {:.4f}".format(confidence))


def get_pytorch_dnn_prediction(original_net, preproc_img, imagenet_labels):
    original_net.eval()
    preproc_img = torch.FloatTensor(preproc_img)

    # inference
    with torch.no_grad():
        out = original_net(preproc_img)

    print("\nPyTorch model prediction: \n")
    print("* shape: ", out.shape)

    # get the predicted class ID
    imagenet_class_id = torch.argmax(out, axis=1).item()
    print("* class ID: {}, label: {}".format(imagenet_class_id, imagenet_labels[imagenet_class_id]))

    # get confidence
    confidence = out[0][imagenet_class_id]
    print("* confidence: {:.4f}".format(confidence.item()))

#end repo files

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    #input_img = get_preprocessed_img(frame)
    #image_label = get_imagenet_labels("classes.csv")

    print(image.shape)
    blob = cv.dnn.blobFromImages(image, size=(640, 480))

#ERRRORSS
#we fixed this error by making the COLOR_ above to gray
#(-215:Assertion failed) total(srcShape, srcRange.start, srcRange.end) == maskTotal in function 'computeShapeByReshapeMask'

#but then didn't fix this one yet: (-2:Unspecified error) Number of input channels should be multiple of 3 but got 640 in function 'getMemoryShapes'

    print(blob.shape)
    opencv_net.setInput(blob)
    opencv_net.forward() #where the error occurs
    
    #get_opencv_dnn_prediction(opencv_net, input_img, image_label)
    
      
    cv.imshow('frame',image)

	#old open sv stuff
    #for (x,y,w,h) in dogs:
        #cv.rectangle(image,(x,y),(x+w,y+h),(416,0,0),2)
        #dog = image[y:y+h, x:x+w]

	#send image
    #rc,png = cv.imencode('.png', image)
    #msg = png.tobytes()
    #client.publish(MQTT_TOPIC, msg, qos=1, retain = False)
    #print('dog', dog)


    print('message sent')
        
        
        # close connection
    if cv.waitKey(1) & 0xFF == ord('q'):
       break


