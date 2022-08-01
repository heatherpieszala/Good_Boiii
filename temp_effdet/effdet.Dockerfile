FROM nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.9-py3

# tested on Jetson NX

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

RUN apt update && apt install -y  libssl-dev
RUN pip3 install -U pip
# Copy contents
# COPY . /usr/src/app
RUN git clone https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch.git 

WORKDIR /usr/src/app/Yet-Another-EfficientDet-Pytorch

# Install dependencies (pip or conda)
# RUN pip3 install -r requirements.txt


RUN apt update && apt install -y libffi-dev python3-pip curl unzip python3-tk libopencv-dev python3-opencv 
RUN pip3 install -U gsutil pyyaml tqdm cython #torchvision   
RUN apt install -y python3-scipy python3-matplotlib python3-numpy
RUN pip3 install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI

# RUN pip3 install requests
# RUN apt install -y python3-pandas
# RUN pip3 install seaborn
RUN pip3 install -U pip
RUN pip list
RUN pip install webcolors
#RUN pip3 install -r requirements.txt

COPY detect_eff.py /usr/src/app/Yet-Another-EfficientDet-Pytorch/detect_eff.py
COPY best.pth /usr/src/app/Yet-Another-EfficientDet-Pytorch/best.pth

CMD ["export", "DISPLAY=:0"]
CMD ["export", "QT_DEBUG_PLUGINS=1"]
CMD ["xhost", "+local:root"]
CMD ["python3","detect_eff.py"]