# Good_Boiii

### Abstract
Ever wonder what your dog is doing when you are not watching?  What about when you are not home?  You might say, I have a camera for that. However, does your camera know when your dog is misbehaving and encourage it to engage in positive behavior instead?  Most likely, it does not. You might also know your dog misbehaves when you leave the room though it is difficult to determine when, and therefore,difficult to successfully work on training it otherwise. That is what our team set out to do and why we wanted to do it.  However, with time constraints and limited access to negative dog behavior data, we decided to focus on the positive behavior portion of the pipeline first.  For the purposes of this analysis, positive behavior means: sit, lay, stand, and play. The following study outlines the approach taken to solve this problem.

## Running the Pipeline
To run this, it is assumed that both docker and kubernetes are already installed on both a Jetson device and cloud VM, and that a cloud instance has already been provisioned.  As part of the Jetson set-up, we already had docker but had to install kubernetes.

## The Jetson
The first step we take is to create a MQTT broker. This process is started by building a docker file for the MQTT broker.  The container is an alpine image and it can be used both on the Jetson and in the cloud.  Here, the cloud accounts we refer to are AWS.
The image is built using the docker build command, then both a mosquitto service and mosquitto deployment are created using kubernetes.
The intent is to have each part of the application connected to and running on kubernetes.  The kubernetes mosquitto service we build here is for our Jetson. 
Once we have a VM running, we run the same commands (and use the same files) to build another service on the cloud VM (t2.micro) which we use to send photos of our dogs to s3 buckets on the cloud.  

```
docker build -t mosquitto:v1 -f Dockerfile.broker .
kubectl apply -f mosquittoService.yaml
kubectl apply -f mosquitto.yaml
```

Now that we have the service running, we can launch the other key pieces of the pipeline. Since detecting whether a dog is sitting or not is core to our use case, we will start here.
We first have to build a docker image with the required package to both capture images from the camera, and to perform the necessary inference on our Jetson device. 
The python file dog_detector.py enables our pipeline to capture the image and inference the action the dog is performing. 
We pursued multiple options for inferencing our models on the Jetson, though settled with two pipeline options.  One uses the built in model from Roboflow. We had used Roboflow to label our images.  The other is a detector that uses our model.  To run either the Roboflow inference pipeline or the teams model, navigate the appropriate folder.  A deployment is built for the dog detector and connected to the mosquito service via port 1883 (it does not matter which dog detector version is run for this).

*Navigate to dog_detector_roboflow and run the following commands to use the Roboflow model.  Navigate to dog_detector_ourmodel to use our model and run the following commands.
```
docker build -t dogcam:v1 -f Dockerfile.dogdetect .
kubectl apply -f dog_detector.yaml
```

In order to recognize if the dog is sitting and reward it, we need to pass along the message that sitting occured. We publish these messages to the cloud through the the MQTT message forwarder.
For this, we need to connect our MQTTlistener within our mosquitto service (the first piece we set up).  This will get the application sending and receiving messages locally.  To connect to the cloud, we need to build a broker image and kubernetes service and deployment for the broker in the cloud, and then connect to the node port of that service within our forwarder.  We also need to cloud VM/instance IP address. This allows us to forward the messages from our Jetson to the cloud.  
```
docker build -t listener:v1 -f Dockerfile.listener .
kubectl apply -f listener.yaml

```
For both the publisher and forwarder, logging is built into the files.  

## The Cloud VM 
SSH into a Bastian Host vitual machine.  A t2.medium can be used.  Ensure that docker and kubernetes are installed - install them if they are not.
If not already set up, set up a MQTT broker on the cloud using a similar methodology as was used on the Jetson above.  Repeat this step, just on the cloud VM.
```
docker build -t mosquitto:v1 -f Dockerfile.broker .
kubectl apply -f mosquittoService.yaml
kubectl apply -f mosquitto.yaml
```
We now have a mosquitto service on the cloud, the node port which should be used in the forwarder (on the Jetson) to send messages to this service.

An image processor was also required to capture the images and send them to an s3 bucket. The details are in dog_listener.py as this is another version of a listener.
The application was deployed using the following and connected to the cloud mosquitto service.
Note that AWS credentials (key and secrect id) are required to connect to s3.  These are removed from the files for the purposes of pushing the code to github and replaced with 'XXX'. This is within the cloud_listener.py file and will need to be populated for the pipeline to run.
```
docker build -t cloud:v1 -f Dockerfile.cloud .
kubectl apply -f cloud_deployment.yaml
```
