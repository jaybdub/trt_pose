#!/bin/bash

docker run --device /dev/video0 \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix \
    --gpus all -it --network host --name trt_pose --rm -v $(pwd):/trt_pose trt_pose:jp50
