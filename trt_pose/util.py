import torch
import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided


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


def coco_category_to_topology(coco_category):
    """Gets topology tensor from a COCO category"""
    skeleton = coco_category["skeleton"]
    K = len(skeleton)
    topology = torch.zeros((K, 4)).int()
    for k in range(K):
        topology[k][0] = 2 * k
        topology[k][1] = 2 * k + 1
        topology[k][2] = skeleton[k][0] - 1
        topology[k][3] = skeleton[k][1] - 1
    return topology


# class DrawObjects(object):
#     def __init__(self, topology):
#         self.topology = topology

#     def __call__(self, image, object_counts, objects, normalized_peaks):
#         topology = self.topology
#         height = image.shape[0]
#         width = image.shape[1]

#         K = topology.shape[0]
#         count = int(object_counts[0])
#         K = topology.shape[0]
#         for i in range(count):
#             color = (0, 255, 0)
#             obj = objects[0][i]
#             C = obj.shape[0]
#             for j in range(C):
#                 k = int(obj[j])
#                 if k >= 0:
#                     peak = normalized_peaks[0][j][k]
#                     x = round(float(peak[1]) * width)
#                     y = round(float(peak[0]) * height)
#                     cv2.circle(image, (x, y), 3, color, 2)

#             for k in range(K):
#                 c_a = topology[k][2]
#                 c_b = topology[k][3]
#                 if obj[c_a] >= 0 and obj[c_b] >= 0:
#                     peak0 = normalized_peaks[0][c_a][obj[c_a]]
#                     peak1 = normalized_peaks[0][c_b][obj[c_b]]
#                     x0 = round(float(peak0[1]) * width)
#                     y0 = round(float(peak0[0]) * height)
#                     x1 = round(float(peak1[1]) * width)
#                     y1 = round(float(peak1[0]) * height)
#                     cv2.line(image, (x0, y0), (x1, y1), color, 2)
