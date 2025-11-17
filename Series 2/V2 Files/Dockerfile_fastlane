FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

RUN apt-get update && apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /bench

# Usage:
#   docker build -t fastlane-bench .
#   docker run --gpus all -it -v $PWD:/bench fastlane-bench /bin/bash
#
# Inside the container:
#   nvcc -O3 -std=c++17 -arch=sm_80 fastlane_best_rr_allinone.cu -lcublasLt -lcublas -o fastlane_best_rr_allinone
#   ./fastlane_best_rr_allinone
