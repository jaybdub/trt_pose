#!/bin/bash

docker run --device /dev/video0 \
    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix \
    --gpus all -it --network host --name trt_pose_prebuilt \
     --rm \
    -w /root/trt_pose/tasks/human_pose \
    trt_pose:jp50-orin-prebuilt python3 demo.py