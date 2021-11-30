from pickle import FALSE
import sys
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import PIL

sys.path.append("./acllite")
from acllite.acllite_resource import AclLiteResource
from acllite.acllite_model import AclLiteModel

IMG_PATH = "images/hand4.jpg"
MODEL_PATH = "model/trt.om"
threshold = 0.03  # 0.1, 0.3
HEATMAP = False

acl_resource = AclLiteResource()
acl_resource.init()
model = AclLiteModel(MODEL_PATH)


def preprocess(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255
    image = image - np.array((0.485, 0.456, 0.406))
    image = image / np.array((0.229, 0.224, 0.225))
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32)
    return image.copy()[None, ...]


def postprocess(heatmap, orig_shape):
    peaks = np.zeros((21, 2))
    for i in range(21):
        heat = heatmap[i, :, :]
        heat = gaussian_filter(heat, sigma=3)
        peak = np.unravel_index(heat.argmax(), heat.shape)
        if heat[peak] > threshold or True:
            peaks[i] = peak
    return peaks * (orig_shape[0] / heatmap.shape[1], orig_shape[1] / heatmap.shape[2])


img = preprocess(IMG_PATH)
ori_img = cv2.imread(IMG_PATH)

cmap, paf = model.execute([img])

with open("trt_pose/tmp/hand4", "wb") as f:
    np.save(f, cmap[0])
    np.save(f, paf[0])

input("")

heatmap = cmap[0]
if HEATMAP:
    for i in range(heatmap.shape[0]):
        plt.imshow(heatmap[i, :, :])
        plt.savefig(f"outputs/heatMat{i}.png")
peaks = postprocess(heatmap, ori_img.shape)
print(peaks)
out = ori_img.copy()
for i, peak in enumerate(peaks):
    position = (round(peak[1]), round(peak[0]))
    out = cv2.circle(
        out,
        position,
        radius=4,
        color=(0, 0, 255),
        thickness=-1,
    )
    out = cv2.putText(
        out,
        str(i),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
cv2.imwrite("outputs/out.jpg", out)
