FROM ubuntu
  
ENV PYTHONUNBUFFERED = 1
WORKDIR /app
RUN apt-get update
RUN apt install -y python3-dev python3-pip
RUN pip3 install paho-mqtt
# example from https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
COPY  msgForwarder.py msgForwarder.py
CMD ["python3","msgForwarder.py"]
