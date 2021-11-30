import numpy as np
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure

threshold = 0.1  # 0.1, 0.3


def postprocess(heatmap, orig_shape):
    peaks = np.zeros((21, 2))
    for i in range(21):
        heat = heatmap[i, :, :]
        heat = gaussian_filter(heat, sigma=3)
        peak = np.unravel_index(heat.argmax(), heat.shape)
        if heat[peak] > threshold or True:
            peaks[i] = peak
    return peaks * (orig_shape[0] / heatmap.shape[1], orig_shape[1] / heatmap.shape[2])


def nms(heatmaps, refine=True, win_size=2):
    peaks_list = []
    for heatmap in heatmaps:
        peak_coords = (heatmap > threshold) * maximum_filter(
            heatmap, footprint=generate_binary_structure(2, 1)
        )
        peak_coords = np.array(np.nonzero(peak_coords)[::-1]).T
        peaks = np.zeros(peak_coords.shape[0], 4)
        for i, peak in enumerate(peak_coords):
            if refine:
                x_min, y_min = np.maximum(0, peak - win_size)
                x_max, y_max = np.minimum(
                    np.array(heatmap.T.shape) - 1, peak + win_size
                )
    return peaks_list
