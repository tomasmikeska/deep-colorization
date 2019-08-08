# --------------------------------------------------------------------
# Base image: Ubuntu 18.04, Nvidia CUDA 10.0
# Python 3
# --------------------------------------------------------------------

FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ENV LANG C.UTF-8

RUN rm -rf /var/lib/apt/lists/*
RUN rm -rf /etc/apt/sources.list.d/cuda.list
RUN rm -rf /etc/apt/sources.list.d/nvidia-ml.list

# --------------------------------------------------------------------
# System libs
# --------------------------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential apt-utils ca-certificates wget git

# --------------------------------------------------------------------
# Python
# --------------------------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common python3-pip python3-distutils-extra

RUN echo 'alias python=python3' >> ~/.bashrc
RUN echo 'alias pip=pip3' >> ~/.bashrc
RUN pip3 --no-cache-dir install --upgrade setuptools

# --------------------------------------------------------------------
# OpenCV
# --------------------------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake unzip pkg-config \
        libjpeg-dev libpng-dev libtiff-dev \
        libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
        libxvidcore-dev libx264-dev \
        libatlas-base-dev gfortran \
        libsm6 libxext6 libxrender-dev

# --------------------------------------------------------------------
# Tensorflow
# --------------------------------------------------------------------

RUN pip3 --no-cache-dir install tensorflow-gpu==1.14.0

# --------------------------------------------------------------------
# Config & Cleanup
# --------------------------------------------------------------------

RUN ldconfig
RUN apt-get clean
RUN apt-get autoremove
RUN rm -rf /var/lib/apt/lists/* /tmp/* ~/*

# --------------------------------------------------------------------
# Ports
# --------------------------------------------------------------------

# Custom tensorboard port
EXPOSE 9090
