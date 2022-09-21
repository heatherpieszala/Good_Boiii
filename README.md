# Good_Boiii

### Abstract

Ever wonder what your dog is doing when you are not watching?  What about when you are not home?  You might say, I have a camera for that. However, does your camera know when your dog is misbehaving and encourage it to engage in *positive* behavior instead?  Most likely, it does not. You might also know your dog misbehaves when you leave the room though it is difficult to determine when, and therefore,difficult to successfully work on training it otherwise. That is what our team set out to do and why we wanted to do it.  However, with time constraints and limited access to *negative* dog behavior data, we decided to focus on the *positive* behavior portion of the pipeline first.  For the purposes of this analysis, *positive* behavior means: sit, lay, stand, and play. The following study outlines the approach taken to solve this problem.

##### Product Vision
![alt text](https://github.com/heatherpieszala/Good_Boiii/blob/main/product_vision.png)

To do this, we investigated and trained models to determine a dog's actions (using yolov5s and efficientdet as our baselines), and then used the weights to inference on a Jetson device.  To properly execute the project, we collected a dataset of dogs depicting 120 breeds, and labeled them sitting or not sitting using Roboflow.  Roboflow also provided a model that we tried. After detecting the dog on the Jetson, and determining the dog's action using our model's weights, we sent the image to an s3 bucket on the cloud if the dog was sitting.  The real motivation behind this would be to have another device dispense a treat if the dog sits. This piece of the pipeline would call the dog away from "negative" behavior, tell it to engage in a positive action like "sit" (or to at least come to the camera), and then reward it for doing so.

Here is an example image from the s3 with the final result.  For Roboflow, we only sent the image if the dog was identified as sitting.  Images for this pipeline piece stop after image #29: [https://dog-training-good-boiii.s3.us-west-1.amazonaws.com/dog-30.jpg](https://dog-training-good-boiii.s3.us-west-1.amazonaws.com/dog-30.jpg)

Here is an example of an image from the yolov5 model pipeline. For this, we also tried to send images of only "sit": https://yolo-model-results.s3.us-west-1.amazonaws.com/dog-582.jpg
Our yolov5 inference turned out to be very good at detecting the dog, and then accurately labeling the appropriate behavior, with at least a low degree of confidence.

##### Architecture
![alt text](https://github.com/heatherpieszala/Good_Boiii/blob/main/architecture.png)

## Running the Pipeline
To run this, it is assumed that both docker and kubernetes are already installed on both a Jetson device and cloud VM, and that a cloud instance has already been provisioned.  As part of the Jetson set-up, we already had docker but had to install kubernetes.

## The Jetson
The first step we take is to create a MQTT broker. This process is started by building a docker file for the MQTT broker.  The container is an alpine image and it can be used both on the Jetson and in the cloud.  Here, the cloud accounts we refer to are AWS.
The image is built using the docker build command, then both a mosquitto service and mosquitto deployment are created using kubernetes.
The intent is to have each part of the application connected to and running on kubernetes.  The kubernetes mosquitto service we build here is for our Jetson. 
Once we have a VM running, we run the same commands (and use the same files) to build another service on the cloud VM (t2.micro) which we use to send photos of our dogs to s3 buckets on the cloud.  

Clone the repo on both the Jetson and cloud instance. Navigate to the service folder and run the following commands.
```
docker build -t mosquitto:v1 -f Dockerfile.broker .
kubectl apply -f mosquittoService.yaml
kubectl apply -f mosquitto.yaml
```
### Detection
Now that we have the service running, we can launch the other key pieces of the pipeline. Since detecting whether a dog is sitting or not is core to our use case, we will start here.
We first have to build a docker image with the required package to both capture images from the camera, and to perform the necessary inference on our Jetson device. We explored two options with different models for detecting the dog sitting.  One was using a Roboflow model (we used Roboflow to label our images), the other was using weights from a custom model we trained on the cloud and used the weights to inference on the Jetson.

#### To Use Roboflow Dog Detector
The python file dog_detector.py enables our pipeline to capture the image and inference the action the dog is performing. The detector file is set up so that if the dog is sitting, the message (and image) will be sent to an s3 bucket on the cloud.  We build a docker container to run this file.
A deployment is then built for the dog detector and connected to the mosquito service via port 1883.  

Navigate to `dog_detection_roboflow` and run the following commands.  We now have the piece of our pipeline that captures the dog sitting.
```
docker build -t dogcam:v1 -f Dockerfile.dogdetect .
kubectl apply -f dog_detect.yaml
```

#### To Use Our Custom Dog Detector
We tried various models for inferencing on the Jetson device.  The first set-up uses yolov5. This piece of the pipeline sends all images of the dog to the cloud. Eventually, we want to detect the dog in other poses. Similar to the roboflow version, we run our python detection file through a Docker container, and launch it on a kubernetes deployment.  Only run one detector at a time.
To run this piece, navigate to the folder `dog_detection_ourmodel/yolov5_ourmodel` and run the following commands:
```
docker build -t dogcam:v1 -f Dockerfile.yolov5 .
kubectl apply -f dog_detect.yaml
```

### Forwarder
In order to recognize if the dog is sitting and reward it, we pass along the message that sitting occured. We publish these messages to the cloud through the the MQTT message forwarder.
For this, we connect our MQTTlistener within our mosquitto service (the first piece we set up).  This allows the application to send and receive messages locally.  To connect to the cloud, we will next build a broker image and kubernetes service and deployment for the broker in the cloud, and then connect to the node port of that service within our forwarder.  The node port and ip need to be specified in the python file we run in the listener docker container. 

Navigate to `listener` and run the following commands.  Ensure that you update the nodeport and ip address in the listener.py.  The listener is our mqtt message Forwarder (mqttMsgForwarder).
```
docker build -t listener:v1 -f Dockerfile.lisener .
kubectl apply -f listener.yaml

```
For both the publisher and forwarder, logging is built into the files.  

## The Cloud VM 
SSH into a Bastian Host vitual machine.  A t2.medium instance can be used.  Ensure that docker and kubernetes are installed - install them if they are not.
If not already set up, set up a MQTT broker on the cloud using a similar methodology as was used on the Jetson above.  To do so, navigate to the `service` folder and run the following commands.
```
docker build -t mosquitto:v1 -f Dockerfile.broker .
kubectl apply -f mosquittoService.yaml
kubectl apply -f mosquitto.yaml
```
We now have a mosquitto service on the cloud, the node port which should be used in the forwarder (on the Jetson) to send messages to this service.

### Processor
An image processor is also required to capture the images and send them to an s3 bucket. The details are in dog_cloud.py.
The application was deployed using the following and connected to the cloud mosquitto service.
Note that AWS credentials (key and secrect id) are required to connect to s3.  These are removed from the files for the purposes of pushing the code to github and replaced with 'XXX'. This is within the dog_cloud.py file and the credentials need to be populated (with yours) for the pipeline to run.
Navigate to the `cloud` folder and run the following commands.
```
docker build -t cloud:v1 -f Dockerfile.cloud .
kubectl apply -f cloud_deployment.yaml
```

### Final Output Examples
https://yolo-model-results.s3.us-west-1.amazonaws.com/dog-582.jpg
![alt text](https://github.com/heatherpieszala/Good_Boiii/blob/main/sage_example.jpg)

![alt text(https://github.com/heatherpieszala/Good_Boiii/blob/main/sage-pipeline.jpg)
