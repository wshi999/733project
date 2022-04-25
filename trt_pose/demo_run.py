from pickle import FALSE
import sys
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

sys.path.append("./acllite")
from acllite.acllite_resource import AclLiteResource
from acllite.acllite_model import AclLiteModel

input_video_path = "images/demo.MP4"
MODEL_PATH = "model/trt_prune.om"
threshold = 0.03  # 0.1, 0.3
HEATMAP = False

acl_resource = AclLiteResource()
acl_resource.init()
model = AclLiteModel(MODEL_PATH)


def preprocess(img_cv2):
    image = cv2.resize(img_cv2, (224, 224), interpolation=cv2.INTER_AREA)
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


def draw(out, peaks):
    for i, peak in enumerate(peaks):
        position = (round(peak[1]), round(peak[0]))
        out = cv2.circle(
            out,
            position,
            radius=15,
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
    return out


cap = cv2.VideoCapture(input_video_path)
ret, img_original = cap.read()
img_shape = img_original.shape
cnt = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# total_frames = 10
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "outputs/video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (img_shape[1], img_shape[0]),
)

# out = np.zeros((total_frames, img_shape[0], img_shape[1], img_shape[2]), dtype=np.uint8)

pbar = tqdm(total=total_frames, ncols=80)
while ret and cnt < total_frames:
    img = preprocess(img_original)
    cmap, paf = model.execute([img])
    peaks = postprocess(cmap[0], img_shape)
    img_out = draw(img_original, peaks)
    # out[cnt] = img_out
    # out.write(img_out)
    # cv2.imshow("out", img_out)
    # cv2.waitkey(1)
    pbar.update(1)
    cnt += 1
    ret, img_original = cap.read()
pbar.close()

out.release()

# imgs = [Image.fromarray(img) for img in out]
# imgs[0].save(
#     "outputs/array.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0
# )
