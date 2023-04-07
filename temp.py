import numpy as np 
import cv2
import torch

def generate_data_field(eye_point):
    """eye_point is (x, y) and between 0 and 1"""
    height, width = 224, 224
    x_grid = np.array(range(width)).reshape([1, width]).repeat(height, axis=0)
    y_grid = np.array(range(height)).reshape([height, 1]).repeat(width, axis=1)
    grid = np.stack((x_grid, y_grid)).astype(np.float32)

    x, y = eye_point
    x, y = x * width, y * height

    grid -= np.array([x, y]).reshape([2, 1, 1]).astype(np.float32)
    norm = np.sqrt(np.sum(grid ** 2, axis=0)).reshape([1, height, width])
    # avoid zero norm
    norm = np.maximum(norm, 0.1)
    grid /= norm
    return grid

def i2s(point_i):
    # 输入image坐标，输出sphere坐标
    assert point_i.shape[1] == 2
    n = point_i.shape[0]
    point_s = np.empty((n, 3))
    point_s[:, 0] = np.pi * (0.5 - point_i[:, 1])
    point_s[:, 1] = 2 * np.pi * (1 - point_i[:, 0])
    point_s[:, 2] = np.ones(n)
    return point_s

def s2c(point_s):
    # 输入sphere坐标，输出camera坐标
    assert point_s.shape[1] == 3
    n = point_s.shape[0]
    point_c = np.empty((n, 3))
    point_c[:, 0] = np.cos(point_s[:, 0]) * np.cos(point_s[:, 1])
    point_c[:, 1] = np.cos(point_s[:, 0]) * np.sin(point_s[:, 1])
    point_c[:, 2] = np.sin(point_s[:, 0])
    return point_c

def i2c(point_i):
    # 输入image坐标，输出camera坐标
    return s2c(i2s(point_i))


def ds(p_i, q_i):
    # 输入image坐标系中的两点，输出sphere坐标系中的视线方向
    assert p_i.shape == q_i.shape
    n = p_i.shape[0]
    p_s = i2s(p_i)
    p_c = s2c(p_s)
    q_c = i2c(q_i)
    d_c = q_c - p_c
    phi, lamda = p_s[:, 0], p_s[:, 1]
    x, y, z = d_c[:, 0], d_c[:, 1], d_c[:, 2]
    d_s = np.empty((n, 3))
    d_s[:, 0] = -np.sin(lamda) * x + np.sin(lamda) * y
    d_s[:, 1] = -np.sin(phi) * np.cos(lamda) * x + -np.sin(phi) * np.sin(lamda) * y + np.cos(phi) * z
    d_s[:, 2] = np.cos(phi) * np.cos(lamda) * x + np.cos(phi) * np.sin(lamda) * y + np.sin(phi) * z
    return d_s


if __name__ == "__main__":
    a = torch.Tensor([0.3, 0.7])
    b = i2c(a.unsqueeze(0))
    print(b.squeeze(0))