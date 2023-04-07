import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import cv2
from collections import OrderedDict
from utils import imutils
from torchvision.models import resnet50, resnet18, resnet34, resnet101, resnet152
from multiprocessing import Pool
import torchsnooper
from RepVGG import *

class GDE_network(nn.Module):
    def __init__(self, resnet_feature_dim=1000, fc_feature_dim=32):
        super(GDE_network, self).__init__()
        self.resnet = resnet50(pretrained=True)
        # self.repvgg_b3=create_RepVGG_B3()
        self.fc_headpostion = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, fc_feature_dim)
        )
        self.fc_to_gaze = nn.Sequential(
            nn.Linear(1000 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3, bias=False)
        )
    
    def forward(self, head_image, head_position):
        resnet_feature = self.resnet(head_image)
        # resnet_feature = self.repvgg_b3(head_image)
        fc_feature = self.fc_headpostion(head_position)
        feature = torch.cat([resnet_feature, fc_feature], axis=1)
        gaze_S = self.fc_to_gaze(feature)

        return gaze_S

class DP_network(nn.Module): 
    def __init__(self):
        super(DP_network, self).__init__()

        self.distant_conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 1, kernel_size=1, stride=1),
        )
    
        self.local_conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 1, kernel_size=1, stride=1),
        )
        
    def forward(self, distant_region, distant_gauss, local_region, attention_map):
        input1 = torch.cat((distant_region, distant_gauss), dim=1)
        input2 = torch.cat((local_region, attention_map), dim=1)
        input1 = input1.cuda()
        input2 = input2.cuda()
        output1 = self.distant_conv(input1)
        output2 = self.local_conv(input2)
        return output1, output2

class gde_dp(nn.Module):
    def __init__(self):
        super(gde_dp, self).__init__()
        self.gde = GDE_network()
        load_weight_from_local(self.gde) # 加载参数并且冻结，不参与训练
        for para in self.gde.parameters():
            para.requires_grad = False
        self.dp = DP_network()
        
    def forward(self, head_image, head_position, img_imf):
        # forward输入都是按batch输入，处理图片维度大小不一致，需要for依次处理
        gaze_S = self.gde(head_image, head_position)
        gaze_S = gaze_S.cpu()

        # 都放在gpu中
        imsize, face_center_raw, face_center_norm, scene_path , wh = img_imf["size"], img_imf["face_center_raw"], img_imf["face_center_norm"], img_imf["scene_path"], img_imf["wh"]
        # print(face_center_norm)
        distant_region, distant_gauss, distant_xyxy, local_region, attention_map, local_xyxy = calculate_input_test(gaze_S, imsize, face_center_norm, face_center_raw, scene_path, wh)
        
        # distant_img = distant_region[0].numpy().transpose(1, 2, 0)
        # cv2.imwrite('distant.jpg', distant_img)
        # local_img = local_region[0].numpy().transpose(1, 2, 0)
        # cv2.imwrite('local.jpg', local_img)
        # exit()

        output1, output2 = self.dp(distant_region, distant_gauss, local_region, attention_map)

        return output1, output2, distant_xyxy, local_xyxy

class gazefollow360(nn.Module): 
    def __init__(self):
        super(gazefollow360, self).__init__()
        self.gde_dp = gde_dp()
        load_weight_from_local(self.gde_dp.dp)
        for para in self.gde_dp.dp.parameters():
            para.requires_grad = False

        self.df = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 1, kernel_size=1, stride=1),
        )
    
    def forward(self, head_image, head_position, img_imf):
        # forward 输入都是按batch输入，处理图片维度大小不一致，需要for依次处理
        output1, output2, distant_xyxy, local_xyxy = self.gde_dp(head_image, head_position, img_imf)

        imsize = img_imf["size"]

        input = get_final_input_test(output1, distant_xyxy, output2, local_xyxy, imsize)
        input = input.cuda()

        output = self.df(input)

        return output

def load_weight_from_local(model_instance):
    # device = torch.device('cuda', 0)
    if isinstance(model_instance, GDE_network):
        model_path = "/home/data/tbw_gaze/gaze323/gaze323-gaze-gazefollow360-/model/gde/gde.pt"
        # gde_state_dict = torch.load(model_path, map_location=device)
        gde_state_dict = torch.load(model_path)
        print("> Loaded model parameters from: %s" % model_path)
        print("> {}".format(model_instance.load_state_dict(gde_state_dict["model"])))
    
    elif isinstance(model_instance, DP_network):
        model_path = "/home/data/tbw_gaze/gaze323/log_2023/dp/dp19/epoch_50_weights.pt"
        dp_state_dict = OrderedDict()
        # state_dict = torch.load(model_path, map_location=device)
        state_dict = torch.load(model_path)
        for k, v in state_dict["model"].items():
            if k[:2] == "dp":
                new_k = k[3:]
                dp_state_dict[new_k] = v
        print("> Loaded model parameters from: %s" % model_path)
        print("> {}".format(model_instance.load_state_dict(dp_state_dict)))    

def func_360(i, output1, distant_xyxy, output2, local_xyxy, imsize):
    width, height = int(imsize[0]), int(imsize[1])
    input1, input2 = np.zeros((height, width)), np.zeros((height, width))
    x_min_dis, y_min_dis, x_max_dis, y_max_dis = int(distant_xyxy[0]*width), int(distant_xyxy[1]*height), int(distant_xyxy[2]*width), int(distant_xyxy[3]*height)
    x_min_loc, y_min_loc, x_max_loc, y_max_loc = int(local_xyxy[0]*width), int(local_xyxy[1]*height), int(local_xyxy[2]*width), int(local_xyxy[3]*height)
    input1[y_min_dis:y_max_dis, x_min_dis:x_max_dis] = cv2.resize(output1.cpu().numpy(), (x_max_dis-x_min_dis, y_max_dis-y_min_dis))
    input2[y_min_loc:y_max_loc, x_min_loc:x_max_loc] = cv2.resize(output2.cpu().numpy(), (x_max_loc-x_min_loc, y_max_loc-y_min_loc))
    input1 = np.expand_dims(cv2.resize(input1, (56, 56)), axis=0)
    input2 = np.expand_dims(cv2.resize(input2, (56, 56)), axis=0)
    input = np.concatenate([input1, input2], axis=0)
    return [i, input]

def get_final_input_test(output1_batch, distant_xyxy_batch, output2_batch, local_xyxy_batch, imsize_batch):
    output1_batch = output1_batch.cpu()
    output2_batch = output2_batch.cpu()
    n = len(output1_batch)
    order = np.array(list(range(n)))
    input_list = np.empty((n, 2, 56, 56))
    output1_batch, output2_batch = output1_batch.permute(0, 2, 3, 1), output2_batch.permute(0, 2, 3, 1)

    def Callback(data):
        input_list[data[0]] = data[1]
    
    def err_call_back(err):
        print(f'2出错啦~ error：{str(err)}')

    ctx = torch.multiprocessing.get_context("spawn")
    pool = ctx.Pool(16)
    for i, output1, distant_xyxy, output2, local_xyxy, imsize in zip(order, output1_batch, distant_xyxy_batch, output2_batch, local_xyxy_batch, imsize_batch):
        pool.apply_async(func_360, (i, output1, distant_xyxy, output2, local_xyxy, imsize), callback=Callback, error_callback=err_call_back)
    pool.close()
    pool.join()

    return torch.Tensor(input_list)

def get_final_input(output1_batch, distant_xyxy_batch, output2_batch, local_xyxy_batch, imsize_batch):

    input_list = []
    device = torch.device('cuda', 0)
    output1_batch, output2_batch = output1_batch.permute(0, 2, 3, 1), output2_batch.permute(0, 2, 3, 1)
    
    for output1, distant_xyxy, output2, local_xyxy, imsize in zip(output1_batch, distant_xyxy_batch, output2_batch, local_xyxy_batch, imsize_batch):
        
        width, height = int(imsize[0]), int(imsize[1])
        input1, input2 = np.zeros((height, width)), np.zeros((height, width))

        x_min_dis, y_min_dis, x_max_dis, y_max_dis = int(distant_xyxy[0]*width), int(distant_xyxy[1]*height), int(distant_xyxy[2]*width), int(distant_xyxy[3]*height)
        x_min_loc, y_min_loc, x_max_loc, y_max_loc = int(local_xyxy[0]*width), int(local_xyxy[1]*height), int(local_xyxy[2]*width), int(local_xyxy[3]*height)
        
        input1[y_min_dis:y_max_dis, x_min_dis:x_max_dis] = cv2.resize(output1.cpu().numpy(), (x_max_dis-x_min_dis, y_max_dis-y_min_dis))
        input2[y_min_loc:y_max_loc, x_min_loc:x_max_loc] = cv2.resize(output2.cpu().numpy(), (x_max_loc-x_min_loc, y_max_loc-y_min_loc))
        
        input1 = np.expand_dims(cv2.resize(input1, (224, 224)), axis=0)
        input2 = np.expand_dims(cv2.resize(input2, (224, 224)), axis=0)
        
        input = np.concatenate([input1, input2], axis=0)
        input_list.append(input)
        
    return (torch.Tensor(input_list)).cuda().to(device)

def func_dp(i, pointS_I, imsize, face_center_raw, scene_path, gaze_S, wh):
    pointS_I_x , pointS_I_y = pointS_I[0] * imsize[0], pointS_I[1] * imsize[1]
    img = cv2.imread(scene_path)
    # cv2.imwrite('row.jpg', img)

    distant_region, distant_gauss, distant_xyxy = get_distant_region(img, [pointS_I_x, pointS_I_y], imsize, ws=imsize[0]/4, hs=imsize[1]/4)
    local_region, attention_map, local_xyxy = get_local_region(img, face_center_raw, gaze_S, wh, imsize, wl=imsize[0]/4, hl=imsize[1]/4)
    return [distant_region, distant_gauss, distant_xyxy, local_region, attention_map, local_xyxy, i]

def calculate_input_test(gaze_S_batch, imsize_batch, face_center_norm_batch, face_center_raw_batch, scene_path_batch, wh_batch):
    # 输入都是batch的形式，且都在gpu上
    face_center_norm_C = i2c(face_center_norm_batch)
    gaze_S_batch = np.array(gaze_S_batch)
    pointS_I = gaze_point_image(face_center_norm_C, gaze_S_batch)

    # print(pointS_I)

    n = len(gaze_S_batch)
    order = np.array(list(range(n)))

    distant_region_list = np.empty((n, 56, 56, 3))
    distant_gauss_list = np.empty((n, 56, 56))
    distant_xyxy_list = np.empty((n, 4))
    local_region_list = np.empty((n, 56, 56, 3))
    attention_map_list = np.empty((n, 56, 56))
    local_xyxy_list = np.empty((n, 4))

    def Callback(data):
        idx = data[6]
        distant_region_list[idx] = data[0]
        distant_gauss_list[idx] = data[1]
        distant_xyxy_list[idx] = data[2]
        local_region_list[idx] = data[3]
        attention_map_list[idx] = data[4]
        local_xyxy_list[idx] = data[5]

    def err_call_back(err):
        print(f'1出错啦~ error：{str(err)}')
    
    ctx = torch.multiprocessing.get_context("spawn")
    pool = ctx.Pool(16)
    for i, pointS_I, imsize, face_center_raw, scene_path, gaze_S, wh in zip(order, pointS_I, imsize_batch, face_center_raw_batch, scene_path_batch, gaze_S_batch, wh_batch):
        pool.apply_async(func_dp, (i, pointS_I, imsize, face_center_raw, scene_path, gaze_S, wh), callback=Callback, error_callback=err_call_back)
    pool.close()
    pool.join()

    # distant_img = distant_gauss_list[0] * 255
    # cv2.imwrite('distant.jpg', distant_img)
    # exit()

    return torch.Tensor(distant_region_list).permute(0, 3, 1, 2), \
        torch.Tensor(distant_gauss_list).reshape(-1, 1, 56, 56), \
        torch.Tensor(distant_xyxy_list), \
        torch.Tensor(local_region_list).permute(0, 3, 1, 2), \
        torch.Tensor(attention_map_list).reshape(-1, 1, 56, 56), \
        torch.Tensor(local_xyxy_list)

def calculate_input(gaze_S_batch, imsize_batch, face_center_norm_batch, face_center_raw_batch, scene_path_batch, wh_batch):
    # 输入都是batch的形式，且都在gpu上
    face_center_norm_C = i2c(face_center_norm_batch)
    distant_region_list, distant_guass_list, distant_xyxy_list, local_region_list, attention_map_list, local_xyxy_list = ([] for i in range(6))
    gaze_S_batch = np.array(gaze_S_batch.cpu())
    pointS_I = gaze_point_image(face_center_norm_C, gaze_S_batch)
    for pointS_I, imsize, face_center_raw, scene_path, gaze_S, wh in zip(pointS_I, imsize_batch, face_center_raw_batch, scene_path_batch, gaze_S_batch, wh_batch):
        pointS_I_x , pointS_I_y = pointS_I[0] * imsize[0], pointS_I[1] * imsize[1]
        img = cv2.imread(scene_path)
        distant_region, distant_gauss, distant_xyxy = get_distant_region(img, [pointS_I_x, pointS_I_y], imsize)
        local_region, attention_map, local_xyxy = get_local_region(img, face_center_raw, gaze_S, wh, imsize)
        distant_region_list.append(distant_region)
        distant_guass_list.append(distant_gauss)
        distant_xyxy_list.append(distant_xyxy)
        local_region_list.append(local_region)
        attention_map_list.append(attention_map)
        local_xyxy_list.append(local_xyxy)

    return torch.Tensor(distant_region_list).reshape(-1, 3, 224, 224), torch.Tensor(distant_guass_list).reshape(-1, 1, 224, 224), torch.Tensor(distant_xyxy_list), torch.Tensor(local_region_list).reshape(-1, 3, 224, 224), torch.Tensor(attention_map_list).reshape(-1, 1, 224, 224), torch.Tensor(local_xyxy_list)

def get_distant_region(img, pointS_I, imsize, ws=600, hs=600, dp_resolution=(56, 56)):
    # 处理单张图片，裁剪出distant_region,并且得到其guass heatmap N，可以考虑将其放入dataset中
    # 注意处理pointS_I是否归一化
    assert len(pointS_I) == 2     
    distant_width, distant_height = int(ws), int(hs)
    x_min, x_max = max(0, int(pointS_I[0] - distant_width/2)), min(int(pointS_I[0] + distant_width/2), int(imsize[0]))
    y_min, y_max = max(0, int(pointS_I[1] - distant_height/2)), min(int(pointS_I[1] + distant_height/2), int(imsize[1]))
    distant_region = img[y_min:y_max, x_min:x_max]

    h, w = distant_region.shape[0], distant_region.shape[1] # 裁剪出region的宽高，防止取整过程中导致宽高不一致
    
    # 先送h再送w
    distant_gauss = torch.zeros([h, w])
    distant_gauss = imutils.draw_labelmap(distant_gauss, [int(w/2), int(h/2)], sigma=15, type='Gaussian') # type is tensor

    # distant_img = distant_gauss.numpy() * 255
    # cv2.imwrite('distant.jpg', distant_img)
    
    distant_region = cv2.resize(distant_region, dp_resolution)
    distant_gauss = cv2.resize(distant_gauss.numpy(), dp_resolution)

    x_min, y_min, x_max, y_max = x_min/imsize[0], y_min/imsize[1], x_max/imsize[0], y_max/imsize[1] 
    return distant_region, distant_gauss, [x_min, y_min, x_max, y_max]   # type is numpy array

def get_local_region(img, pointP_I, d_s, wh, imsize, wl=600, hl=600, dp_resolution=(56, 56)):
    # wh为输入的head of the human subject, wl, hl的值与wh的值有关, 由超参数设置
    # 裁剪出local_region,并且得到其attention map A_l
    # 其实d_s是S坐标下的P到Q的sight line
    # 注意d_s_x等值是否归一化之后
    assert len(d_s) == 3
    d_s_x, d_s_y = d_s[0], d_s[1]
    local_width, local_height = int(wl), int(hl)
    y_min, y_max = max(0, int(pointP_I[1] - local_height/2)), min(int(pointP_I[1] + local_height/2), int(imsize[1]))
    center_x = int(pointP_I[0] - np.sign(d_s_x) * (wl - wh)/2)
    x_min, x_max = max(0, int(center_x - local_width/2)), min(int(center_x + local_width/2), int(imsize[0]))
    local_region = img[y_min:y_max, x_min:x_max]
    h, w = local_region.shape[0], local_region.shape[1]
    
    attention_map = torch.zeros([h, w])

    attention_map = imutils.draw_labelmap(attention_map, [int(w/2), int(h/2)], sigma=15, type='Gaussian') # type is tensor
    
    local_region = cv2.resize(local_region, dp_resolution)
    attention_map = cv2.resize(attention_map.numpy(), dp_resolution)
    x_min, y_min, x_max, y_max = x_min/imsize[0], y_min/imsize[1], x_max/imsize[0], y_max/imsize[1]
    return local_region, attention_map, [x_min, y_min, x_max, y_max]

def I2C(point_i):
    # 输入I坐标系下，输出C坐标系下
    # point_I以batch形式输入
    assert point_i.shape[1] == 2
    n = point_i.shape[0]
    point_c = np.empty((n, 3))
    point_c[:, 0] = np.multiply(np.sin(np.pi * point_i[:, 1]), np.cos(2*np.pi * point_i[:, 0]))
    point_c[:, 1] = -np.multiply(np.sin(np.pi * point_i[:, 1]), np.sin(2*np.pi * point_i[:, 0]))
    point_c[:, 2] = np.cos(np.pi * point_i[:, 1])
    return point_c

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
    d_s[:, 0] = -np.sin(lamda) * x + np.cos(lamda) * y
    d_s[:, 1] = -np.sin(phi) * np.cos(lamda) * x + -np.sin(phi) * np.sin(lamda) * y + np.cos(phi) * z
    d_s[:, 2] = np.cos(phi) * np.cos(lamda) * x + np.cos(phi) * np.sin(lamda) * y + np.sin(phi) * z
    return d_s

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

def relu(a):
    return(np.maximum(0, a))

def gaze_point(point_c, d_s):
    # 输入输入camera坐标系中的头部位置和subject坐标系中的视线方向，输出cemara坐标系中的注视点
    assert point_c.shape == d_s.shape
    n = point_c.shape[0]
    gaze = np.empty((n, 3))
    d_c = dc(point_c, d_s)
    x_p, y_p, z_p = point_c[:, 0], point_c[:, 1], point_c[:, 2]
    x_d, y_d, z_d = d_c[:, 0], d_c[:, 1], d_c[:, 2]
    # matlab求解方程得到解析解
    gaze[:, 0] = (x_p * z_d - x_d * z_p + (x_d * (z_d * np.sqrt(relu(-np.square(x_d) * np.square(y_p) - np.square(x_d) * np.square(z_p) + np.square(x_d) + 2 * x_d * x_p * y_d * y_p + 2 * x_d * x_p * z_d * z_p - np.square(x_p) * np.square(y_d) - np.square(x_p) * np.square(z_d) - np.square(y_d) * np.square(z_p) + np.square(y_d) + 2 * y_d * y_p * z_d * z_p - np.square(y_p) * np.square(z_d) + np.square(z_d))) + np.square(x_d) * z_p + np.square(y_d) * z_p - x_d * x_p * z_d - y_d * y_p * z_d))/(np.square(x_d) + np.square(y_d) + np.square(z_d)))/z_d
    gaze[:, 1] = (y_p * z_d - y_d * z_p + (y_d * (z_d * np.sqrt(relu(-np.square(x_d) * np.square(y_p) - np.square(x_d) * np.square(z_p) + np.square(x_d) + 2 * x_d * x_p * y_d * y_p + 2 * x_d * x_p * z_d * z_p - np.square(x_p) * np.square(y_d) - np.square(x_p) * np.square(z_d) - np.square(y_d) * np.square(z_p) + np.square(y_d) + 2 * y_d * y_p * z_d * z_p - np.square(y_p) * np.square(z_d) + np.square(z_d))) + np.square(x_d) * z_p + np.square(y_d) * z_p - x_d * x_p * z_d - y_d * y_p * z_d))/(np.square(x_d) + np.square(y_d) + np.square(z_d)))/z_d
    gaze[:, 2] = (z_d * np.sqrt(relu(-np.square(x_d) * np.square(y_p) - np.square(x_d) * np.square(z_p) + np.square(x_d) + 2 * x_d * x_p * y_d * y_p + 2 * x_d * x_p * z_d * z_p - np.square(x_p) * np.square(y_d) - np.square(x_p) * np.square(z_d) - np.square(y_d) * np.square(z_p) + np.square(y_d) + 2 * y_d * y_p * z_d * z_p - np.square(y_p) * np.square(z_d) + np.square(z_d))) + np.square(x_d) * z_p + np.square(y_d) * z_p - x_d * x_p * z_d - y_d * y_p * z_d)/(np.square(x_d) + np.square(y_d) + np.square(z_d))
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

if __name__ == "__main__":
    img = cv2.imread('/home/data/rl/2023/gazefollow360/img_aug.jpg')
    print(img.shape)


    
    
    
    
    


