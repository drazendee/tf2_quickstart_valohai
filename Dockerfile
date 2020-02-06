# We'll use the nvidia/cuda image as our base
FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

# Set some common environmenta variables that Python uses
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install lower level dependencies
# Run newest updates, install python etc.
RUN apt-get update --fix-missing && \
    apt-get install -y curl python3 python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    apt install -y libsm6 libxext6 libxrender-dev && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Define our working directory
WORKDIR  /usr/src/valohai-tf2-quickstart

# Installing python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
