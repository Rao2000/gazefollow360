import torch
from model import GDE_network, DP_network, gde_dp
import numpy as np
from torch import nn
import os
from collections import OrderedDict
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def load_weight_from_local(model_instance):
    device = torch.device('cuda', 0)
    if isinstance(model_instance, GDE_network):
        model_path = "/home/data/rl/2023/gazefollow360_output/log1/epoch_50_weights.pt"
        gde_state_dict = torch.load(model_path, map_location=device)
        for k, v in gde_state_dict["model"].items():
            print(k)
        print("> Loaded model parameters from: %s" % model_path)
        print("> {}".format(model_instance.load_state_dict(gde_state_dict["model"])))
    
    elif isinstance(model_instance, DP_network):
        model_path = "/home/data/rl/2023/gazefollow360_output/dp_log1/epoch_16_weights.pt"
        dp_state_dict = OrderedDict()
        state_dict = torch.load(model_path, map_location=device)
        for k, v in state_dict["model"].items():
            if k[:2] == "dp":
                new_k = k[3:]
                dp_state_dict[new_k] = v
            print(k)
        print("> Loaded model parameters from: %s" % model_path)
        print("> {}".format(model_instance.load_state_dict(dp_state_dict)))    

def c2s(point_c):
    # 输入camera坐标，输出sphere坐标
    assert point_c.shape[1] == 3
    n = point_c.shape[0]
    point_s = np.empty((n, 3))
    # 计算phi
    # arcsin中的数的绝对值应小于等于1
    point_s[:, 0] = np.arcsin(point_c[:, 2])
    # 计算lamda
    # x=0 y>0 lamda为pi/2
    # x=0 y<0 lamda为3pi/2
    # x=0 y=0 lamda为0
    idx_x_zero_y_positive = np.array(list(set(np.where(point_c[:, 0] == 0)[0]) & set(np.where(point_c[:, 1] > 0)[0]))).astype('int64')
    idx_x_zero_y_negative = np.array(list(set(np.where(point_c[:, 0] == 0)[0]) & set(np.where(point_c[:, 1] < 0)[0]))).astype('int64')
    idx_x_zero_y_zero = np.array(list(set(np.where(point_c[:, 0] == 0)[0]) & set(np.where(point_c[:, 1] == 0)[0]))).astype('int64')
    idx_x_positive_y_positive = np.array(list(set(np.where(point_c[:, 0] > 0)[0]) & set(np.where(point_c[:, 1] > 0)[0]))).astype('int64')
    idx_x_positive_y_negative = np.array(list(set(np.where(point_c[:, 0] > 0)[0]) & set(np.where(point_c[:, 1] < 0)[0]))).astype('int64')
    idx_x_negative_y_positive = np.array(list(set(np.where(point_c[:, 0] < 0)[0]) & set(np.where(point_c[:, 1] > 0)[0]))).astype('int64')
    idx_x_negative_y_negative = np.array(list(set(np.where(point_c[:, 0] < 0)[0]) & set(np.where(point_c[:, 1] < 0)[0]))).astype('int64')
    point_s[idx_x_positive_y_positive, 1] = np.arctan(point_c[idx_x_positive_y_positive, 1] / point_c[idx_x_positive_y_positive, 0])
    point_s[idx_x_negative_y_positive, 1] = np.arctan(point_c[idx_x_negative_y_positive, 1] / point_c[idx_x_negative_y_positive, 0]) + np.pi
    point_s[idx_x_negative_y_negative, 1] = np.arctan(point_c[idx_x_negative_y_negative, 1] / point_c[idx_x_negative_y_negative, 0]) + np.pi
    point_s[idx_x_positive_y_negative, 1] = np.arctan(point_c[idx_x_positive_y_negative, 1] / point_c[idx_x_positive_y_negative, 0]) + 2 * np.pi
    point_s[idx_x_zero_y_positive, 1] = np.pi / 2
    point_s[idx_x_zero_y_negative, 1] = 3 * np.pi / 2
    point_s[idx_x_zero_y_zero, 1] = 0
    # 计算r 归一化
    point_s[:, 2] = 1
    return point_s

def dc(point_c, d_s):
    # 输入camera坐标系中的头部位置和subject坐标系中的视线方向，输出cemara坐标系中的视线方向
    assert point_c.shape == d_s.shape
    n = point_c.shape[0]
    d_c = np.empty((n, 3))
    point_s = c2s(point_c)
    phi, lamda = point_s[:, 0], point_s[:, 1]
    x, y, z = d_s[:, 0], d_s[:, 1], d_s[:, 2]
    # 变换矩阵R为正交矩阵，其转置和逆相同
    d_c[:, 0] = -np.sin(lamda) * x + -np.sin(phi) * np.cos(lamda) * y + np.cos(phi) * np.cos(lamda) * z
    d_c[:, 1] = np.cos(lamda) * x + -np.sin(phi) * np.sin(lamda) * y + np.cos(phi) * np.sin(lamda) * z
    d_c[:, 2] = np.cos(phi) * y + np.sin(phi) * z
    return d_c    

def gaze_point(point_c, d_s):
    # 输入输入camera坐标系中的头部位置和subject坐标系中的视线方向，输出cemara坐标系中的注视点
    assert point_c.shape == d_s.shape
    n = point_c.shape[0]
    gaze = np.empty((n, 3))
    d_c = dc(point_c, d_s)
    # print("d_c = ", d_c)
    x_p, y_p, z_p = point_c[:, 0], point_c[:, 1], point_c[:, 2]
    x_d, y_d, z_d = d_c[:, 0], d_c[:, 1], d_c[:, 2]
    # matlab求解方程得到解析解
    gaze[:, 0] = (x_p*z_d - x_d*z_p + (x_d*(z_d*(- x_d**2*y_p**2 - x_d**2*z_p**2 + x_d**2 + 2*x_d*x_p*y_d*y_p + 2*x_d*x_p*z_d*z_p - x_p**2*y_d**2 - x_p**2*z_d**2 - y_d**2*z_p**2 + y_d**2 + 2*y_d*y_p*z_d*z_p - y_p**2*z_d**2 + z_d**2)**(1/2) + x_d**2*z_p + y_d**2*z_p - x_d*x_p*z_d - y_d*y_p*z_d))/(x_d**2 + y_d**2 + z_d**2))/z_d
    gaze[:, 1] = (y_p*z_d - y_d*z_p + (y_d*(z_d*(- x_d**2*y_p**2 - x_d**2*z_p**2 + x_d**2 + 2*x_d*x_p*y_d*y_p + 2*x_d*x_p*z_d*z_p - x_p**2*y_d**2 - x_p**2*z_d**2 - y_d**2*z_p**2 + y_d**2 + 2*y_d*y_p*z_d*z_p - y_p**2*z_d**2 + z_d**2)**(1/2) + x_d**2*z_p + y_d**2*z_p - x_d*x_p*z_d - y_d*y_p*z_d))/(x_d**2 + y_d**2 + z_d**2))/z_d
    gaze[:, 2] = (z_d*(- x_d**2*y_p**2 - x_d**2*z_p**2 + x_d**2 + 2*x_d*x_p*y_d*y_p + 2*x_d*x_p*z_d*z_p - x_p**2*y_d**2 - x_p**2*z_d**2 - y_d**2*z_p**2 + y_d**2 + 2*y_d*y_p*z_d*z_p - y_p**2*z_d**2 + z_d**2)**(1/2) + x_d**2*z_p + y_d**2*z_p - x_d*x_p*z_d - y_d*y_p*z_d)/(x_d**2 + y_d**2 + z_d**2)
    return gaze

def c2i(point_c):
    # 输入camera坐标，输出image坐标
    assert point_c.shape[1] == 3
    n = point_c.shape[0]
    point_i = np.empty((n, 2))
    point_s = c2s(point_c)
    # print(point_s)
    phi = point_s[:, 0]
    lamda = point_s[:, 1]
    point_i[:, 0] = 1 - lamda / (2 * np.pi)
    point_i[:, 1] = 0.5 - phi / np.pi
    return point_i

def gaze_point_image(point_c, d_s):
    # 输入输入camera坐标系中的头部位置和subject坐标系中的视线方向，输出image坐标系中的注视点
    return c2i(gaze_point(point_c, d_s))

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.cov1 = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.con2 = nn.Conv2d(32, 256, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.ct1 = nn.ConvTranspose2d(256, 32, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.ct2 = nn.ConvTranspose2d(32, 4, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(4)
        self.con3 = nn.Conv2d(4, 1, kernel_size=1, stride=1)
    def forward(self, x):
        x = self.cov1(x)
        print(x.shape)
        x = self.bn1(x)
        print(x.shape)
        x = self.con2(x)
        print(x.shape)
        x = self.bn2(x)
        print(x.shape)
        x = self.ct1(x)
        print(x.shape)
        x = self.bn3(x)
        print(x.shape)
        x = self.ct2(x)
        print(x.shape)
        x = self.bn4(x)
        print(x.shape)
        x = self.con3(x)
        print(x.shape)
        return x

if __name__ == "__main__":
    dp = gde_dp()
    print(dp.dp)
    load_weight_from_local(dp.dp)
    
