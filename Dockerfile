#FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
#FROM nvidia/cudagl:11.4.1-runtime-ubuntu20.04
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=TZ=Etc/UTC
RUN apt-get update && apt-get install -y \
  git apt-utils python3-pip libgl1-mesa-glx libglib2.0-0

# Install python packages
# RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN git clone https://github.com/xinntao/Real-ESRGAN.git
# we picked our own versions in our own requirements.txt
# RUN cd Real-ESRGAN && pip3 install -r requirements.txt && python3 setup.py develop
RUN cd Real-ESRGAN && python3 setup.py develop

# We add the banana boilerplate here
ADD server.py .
EXPOSE 8000

ADD models.py .

#COPY weights weights
ADD download.py .
RUN python3 download.py

# Add your custom app code, init() and inference()
ADD send.py .
ADD app.py .
RUN python3 app.py

CMD python3 -u server.py
