import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from util import coco_category_to_topology
from numpy.lib.stride_tricks import as_strided
from model import resnet18_baseline_att
import numpy as np
import torch
import PIL
import torchvision

MODELPATH = "model/hand_pose_resnet18_att_244_244.pth"
IMG = "images/hand.jpg"

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


def pool2d(A, kernel_size, stride, padding, pool_mode="max"):
    """
    2D Pooling
    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    """
    # Padding
    A = np.pad(A, padding, mode="constant")

    # Window view of A
    output_shape = (
        (A.shape[0] - kernel_size) // stride + 1,
        (A.shape[1] - kernel_size) // stride + 1,
    )
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(
        A,
        shape=output_shape + kernel_size,
        strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides,
    )
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == "max":
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == "avg":
        return A_w.mean(axis=(1, 2)).reshape(output_shape)


def nms(heatmaps):
    results = np.empty_like(heatmaps)
    for i in range(heatmaps.shape[0]):
        heat = heatmaps[i, :, :]
        hmax = pool2d(heat, 3, 1, 1)
        keep = (hmax == heat).astype(float)

        results[i, :, :] = heat * keep
    return results


topology = coco_category_to_topology(hand_pose)
num_parts = len(hand_pose["keypoints"])
num_links = len(hand_pose["skeleton"])
model = resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
model.load_state_dict(torch.load(MODELPATH))
img = cv2.imread(IMG)
img = preprocess(img)
cmap, paf = model(img)

print(cmap.shape, paf.shape)

heatmap = cmap.squeeze().cpu().detach().numpy()
for i in range(heatmap.shape[0]):
    plt.imshow(heatmap[i, :, :])
    plt.savefig(f"outputs/heatMat{i}.png")

peaks = nms(heatmap)
print(peaks.shape)
