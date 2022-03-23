# Base image must at least have pytorch and CUDA installed.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:21.10-py3
FROM $BASE_IMAGE
ARG BASE_IMAGE
RUN pip install numpy==1.21.2 scipy==1.6.3 pandas==1.4.1 matplotlib==3.4.3 librosa==0.8.1 tensorboardX==2.5 h5py==3.6.0 tqdm
WORKDIR /mnt
