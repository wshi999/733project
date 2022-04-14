import numpy as np
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
import cv2

threshold = 0.3  # 0.1, 0.3


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
    cnt_total_joints = 0
    for heatmap in heatmaps:
        peak_coords = (heatmap > threshold) * maximum_filter(
            heatmap, footprint=generate_binary_structure(2, 1)
        )
        peak_coords = np.array(np.nonzero(peak_coords)[::-1]).T
        peaks = np.zeros((peak_coords.shape[0], 4))
        for i, peak in enumerate(peak_coords):
            if refine:
                x_min, y_min = np.maximum(0, peak - win_size)
                x_max, y_max = np.minimum(
                    np.array(heatmap.T.shape) - 1, peak + win_size
                )

                patch = heatmap[y_min : y_max + 1, x_min : x_max + 1]
                map_upsamp = cv2.resize(
                    patch, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_CUBIC
                )

                location_of_max = np.unravel_index(
                    map_upsamp.argmax(), map_upsamp.shape
                )

                location_of_patch_center = compute_resized_coords(
                    peak[::-1] - [y_min, x_min], 1.0
                )
                refined_center = location_of_max - location_of_patch_center
                peak_score = map_upsamp[location_of_max]
            else:
                refined_center = [0, 0]
                peak_score = heatmap[tuple(peak[::-1])]
            peaks[i, :] = tuple(
                [
                    int(round(x))
                    for x in compute_resized_coords(peak_coords[i], 1.0)
                    + refined_center[::-1]
                ]
            ) + (peak_score, cnt_total_joints)
            cnt_total_joints += 1
        peaks_list.append(peaks)
    return peaks_list


def compute_resized_coords(coords, resizeFactor):
    return (np.array(coords, dtype=float) + 0.5) * resizeFactor - 0.5
