import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from util import coco_category_to_topology
from model import resnet18_baseline_att
import numpy as np
import torch
import PIL
import torchvision
from scipy.ndimage.filters import gaussian_filter
import math
import torch.nn as nn

MODELPATH = "model/hand_pose_resnet18_att_244_244.pth"
IMG = "images/hand.jpg"
threshold = 0.03  # 0.1, 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("trt_pose/hand_pose.json", "r") as f:
    hand_pose = json.load(f)

mean = torch.Tensor([0.485, 0.456, 0.406])
std = torch.Tensor([0.229, 0.224, 0.225])


def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = torchvision.transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])

    return image[None, ...]


topology = coco_category_to_topology(hand_pose)
num_parts = len(hand_pose["keypoints"])
num_links = len(hand_pose["skeleton"])
model = resnet18_baseline_att(num_parts, 2 * num_links).eval()
model.load_state_dict(torch.load(MODELPATH, map_location=torch.device("cpu")))

model[0].resnet.avgpool = nn.AvgPool2d(224)

ori_img = cv2.imread(IMG)
ori_img = cv2.resize(ori_img, (224, 224), interpolation=cv2.INTER_AREA)
ori_shape = ori_img.shape[0:2]
img = preprocess(ori_img)
print(img.shape)
cmap, paf = model(img)

torch.onnx.export(
    model,
    img,
    "model/trt_pose.onnx",
    export_params=True,
    output_names=["cmap", "paf"],
    input_names=["image"],
)
