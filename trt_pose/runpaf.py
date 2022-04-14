import numpy as np
from paf import postprocess, nms

with open("trt_pose/tmp/hand4", "rb") as f:
    heatmaps = np.load(f)
    paf = np.load(f)

peaks = postprocess(heatmaps, (1, 1))
print(peaks * heatmaps[0].shape)
peakcoord = nms(heatmaps=heatmaps)
print(peakcoord)
