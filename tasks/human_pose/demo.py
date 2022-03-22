import json
import os
import argparse
import trt_pose.coco
import trt_pose.models
from trt_pose.utils.preprocess import preprocess
from trt_pose.parse_objects import PostProcess
import torch
from torch2trt import torch2trt, TRTModule
import cv2
import torchvision.transforms as transforms
import PIL.Image


parser = argparse.ArgumentParser()
parser.add_argument('--camera_device', type=int, default=0)
parser.add_argument('--task', type=str, default='human_pose.json')
parser.add_argument('--weights', type=str, default='densenet121_baseline_att_256x256_B_epoch_160.pth')
parser.add_argument('--model', type=str, default='densenet121_baseline_att')
parser.add_argument('--model_trt', type=str, default='model_trt.pth')
parser.add_argument('--force_build', action='store_true')
parser.add_argument('--image_shape', type=str, default='256x256')
args = parser.parse_args()


def v4l2_gst_str(device, capture_width=640, capture_height=480, capture_fps=30):
    return 'v4l2src device=/dev/video{} ! video/x-raw, width=(int){}, height=(int){}, framerate=(fraction){}/1 ! videoconvert !  video/x-raw, format=(string)BGR ! appsink'.format(
        device, capture_width, capture_height, capture_fps)


image_shape = tuple([int(d) for d in args.image_shape.split('x')])

with open(args.task, 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)
post_process = PostProcess(args.task)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

model = trt_pose.models.MODELS[args.model](num_parts, 2 * num_links).cuda().eval()
model.load_state_dict(torch.load(args.weights))

data = torch.randn((1, 3, image_shape[0], image_shape[1])).cuda()

if not os.path.exists(args.model_trt) or args.force_build:
    print('Building TensorRT engine.')
    model_trt = torch2trt(model, [data], fp16_mode=True)
    torch.save(model_trt.state_dict(), args.model_trt)
else:
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(args.model_trt))


cap = cv2.VideoCapture(v4l2_gst_str(args.camera_device), cv2.CAP_GSTREAMER)

while True:
    re, image = cap.read()
    image = cv2.resize(image, image_shape)
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    objects = post_process(cmap, paf)
    print(objects)