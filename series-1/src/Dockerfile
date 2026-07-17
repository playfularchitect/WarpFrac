FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
RUN apt-get update && apt-get install -y build-essential libgmp-dev git && rm -rf /var/lib/apt/lists/*
WORKDIR /bench
# copy sources at runtime (mount -v)
# docker run --gpus all -it -v $PWD:/bench image:tag /bin/bash
# then:
# nvcc -O3 -std=c++17 -arch=sm_80 fx_int8_kpanel_tiled_swarm_v1.cu -lcublasLt -lcublas -lgmp -o fx
# ./fx --m 5120 --n 5120 --k 5120 --streams 32 --graphNodes 64 --batchPerNode 4 --tileK 1280 --epochs 1 --warmup 3 --tryAlgos 16 --workspaceMB 1024 --validate 0
