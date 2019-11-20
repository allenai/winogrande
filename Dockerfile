FROM library/python:3.6

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PATH /usr/local/nvidia/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PYTHONPATH .:$PYTHONPATH

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

LABEL com.nvidia.volumes.needed="nvidia_driver"

# set the working directory
WORKDIR winogrande

# Install base packages.
RUN apt-get update --fix-missing && apt-get install -y \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    libevent-dev \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3.6 install -r requirements.txt

ENV PYTHONPATH .

# Download spacy model
RUN spacy download en_core_web_sm

# install python packages
COPY scripts scripts


RUN mkdir /results

# define the default command
# if you need to run a long-lived process, use 'docker run --init'
CMD ["/bin/bash"]
