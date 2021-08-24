#! /bin/bash
WORK_DIR=/data_shared/Docker/uncertainty-baselines/
PORT=8007
docker build -t uncertainty-baselines docker/
docker run -it --name uncertainty-baselines --gpus all \
    -v ${WORK_DIR}:/usr/app \
    -p ${PORT}:8888 \
    --shm-size 16G \
    uncertainty-baselines
