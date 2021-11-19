import sys
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter


sys.path.append("./acllite")
from acllite.acllite_resource import AclLiteResource
from acllite.acllite_model import AclLiteModel

MODEL_PATH = "model/trt_with_paf.om"
IMG_PATH = "images/hand.jpg"
threshold = 0.03  # 0.1, 0.3


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


img = preprocess(IMG_PATH)
ori_img = cv2.imread(IMG_PATH)

cmap, paf = model.execute([img])


heatmap = cmap[0]
peaks = postprocess(heatmap, (224, 224))
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
