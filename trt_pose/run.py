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

MODELPATH = "model/hand_pose_resnet18_att_244_244.pth"
IMG = "images/demo.jpg"
threshold = 0.03  # 0.1, 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("trt_pose/hand_pose.json", "r") as f:
    hand_pose = json.load(f)


# WIDTH = 224
# HEIGHT = 224
# data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()


def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = torchvision.transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


def postprocess(heatmap, orig_shape):
    peaks = np.zeros((21, 2))
    for i in range(21):
        heat = heatmap[i, :, :]
        heat = gaussian_filter(heat, sigma=3)
        peak = np.unravel_index(heat.argmax(), heat.shape)
        if heat[peak] > threshold:
            peaks[i] = peak
    return peaks * (orig_shape[0] / heatmap.shape[1], orig_shape[1] / heatmap.shape[2])


topology = coco_category_to_topology(hand_pose)
num_parts = len(hand_pose["keypoints"])
num_links = len(hand_pose["skeleton"])
model = resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
model.load_state_dict(torch.load(MODELPATH))
ori_img = cv2.imread(IMG)
ori_shape = ori_img.shape[0:2]
img = preprocess(ori_img)
cmap, paf = model(img)

print(cmap.shape, paf.shape)

heatmap = cmap.squeeze().cpu().detach().numpy()
for i in range(heatmap.shape[0]):
    plt.imshow(heatmap[i, :, :])
    plt.savefig(f"outputs/heatMat{i}.png")

peaks = postprocess(heatmap, ori_shape)
print(peaks)
out = ori_img.copy()
for peak in peaks:
    out = cv2.circle(
        out,
        (round(peak[1]), round(peak[0])),
        radius=4,
        color=(0, 0, 255),
        thickness=-1,
    )
cv2.imwrite("outputs/out.jpg", out)
