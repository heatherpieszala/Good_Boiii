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

import torch
from torch.backends import cudnn

from backbone import EfficientDetBackbone
import cv2
import matplotlib.pyplot as plt
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

#could not get torch2trt to instlal on my jetson - aneeded tensorrt and then the whl file wasn't working to resolve error
#from torch2trt import TRTModule

#model_trt = TRTModule()
#model_trt.load_state_dict(torch.load('efficientdet-d0_74_17600.pth'))
#model = torch.load('efficientdet-d0_74_17600.pth')

#direct xml not working with cascade seems like we need to make cascade specific xml
#doggies = cv.CascadeClassifier('efficientdet-d0_74_17600.xml')

#primary mode to try - from this link:https://docs.opencv.org/4.x/dc/d70/pytorch_cls_tutorial_dnn_conversion.html
#opencv_net = cv.dnn.readNetFromONNX('best.onnx')

#secondary - found in this post when trying to figure out solution
#opencv_net = cv.dnn_ClassificationModel('best.onnx') come back to this if readNetFromONNX not working with forward and blob

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

            cv.putText(ori_imgs[i], '{}, {:.3f}'.format(obj, score),
                    (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1)

            plt.imshow(ori_imgs[i])

    #input_img = get_preprocessed_img(frame)
    #image_label = get_imagenet_labels("classes.csv")

    #print(image.shape)
    #blob = cv.dnn.blobFromImages(image, size=(640, 480))

#ERRRORSS
#we fixed this error by making the COLOR_ above to gray
#(-215:Assertion failed) total(srcShape, srcRange.start, srcRange.end) == maskTotal in function 'computeShapeByReshapeMask'

#but then didn't fix this one yet: (-2:Unspecified error) Number of input channels should be multiple of 3 but got 640 in function 'getMemoryShapes'

    #print(blob.shape)
    #opencv_net.setInput(blob)
    #opencv_net.forward() #where the error occurs
    
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


