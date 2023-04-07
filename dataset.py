import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image
import pandas as pd
import torchvision
import scipy.io as sio

import matplotlib as mpl

mpl.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt

import os

from utils import imutils
from utils import myutils
from config import *
from model import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def loadMetadata(filename, silent = False):
    try:
        # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
        if not silent:
            print('\tReading metadata from %s...' % filename)
        metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
    except:
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata

class GazeFollow360(Dataset): # gazefollow360
    def __init__(self, root_path, split='train', imshow=False, part='gde'):
        assert split in ['train', 'test', 'vali']
        print('Loading GazeFollow360 dataset...')
        metaFile = os.path.join(root_path, 'train_vali_test_data', '{}_data.mat'.format(split))
        if metaFile is None or not os.path.isfile(metaFile):
            raise RuntimeError('There is no such file %s! Provide a valid dataset path.' % metaFile)
        self.metadata = loadMetadata(metaFile)
        if self.metadata is None:
            raise RuntimeError('Could not read metadata file %s! Provide a valid dataset path.' % metaFile)

        self.face_transform = transforms.Compose([
                            transforms.Resize(face_resolution),
                            transforms.ToTensor(), 
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
        self.scene_transform = transforms.Compose([
                            transforms.Resize((scene_resolution[1], scene_resolution[0])),
                            transforms.ToTensor(), 
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])

        self.data_dir = root_path
        self.face_size = face_resolution
        self.scene_size = scene_resolution
        self.output_size = output_resolution
        self.imshow = imshow
        self.part = part
        self.test = (split=="test")

        print('Loaded GazeFollow360 dataset split "%s" with %d records...' % (split, self.metadata['gaze'].shape[0]))

    def __getitem__(self, index):
        # load all information
        scene_path = os.path.join(self.data_dir, "all the videos", str(self.metadata['path'][index]).rstrip()) 
        rec_xyxy = list(map(int,self.metadata['face_box'][index].reshape(-1)))
        gaze_x, gaze_y = self.metadata['gaze'][index].tolist()
        x_min, y_min, x_max, y_max = rec_xyxy
        
        # examine face box
        x_tmp = min(x_min, x_max)
        x_max = max(x_min, x_max)
        x_min = x_tmp
        y_tmp = min(y_min, y_max)
        y_max = max(y_min, y_max)
        y_min = y_tmp

        # expand face bbox a bit
        k = 0.1
        x_min -= k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += k * abs(x_max - x_min)
        y_max += k * abs(y_max - y_min)

        img = Image.open(scene_path)
        img = img.convert('RGB')
        width, height = img.size[:2]
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

        if self.imshow:
            img.save("origin_img.jpg")

        imsize = torch.Tensor([width, height])
        PoG = torch.Tensor([gaze_x, gaze_y])
        
        ## data augmentation
        if is_jitter:
            # Jitter (expansion-only) bounding box size
            if np.random.random_sample() <= 0.5:
                k = np.random.random_sample() * 0.2
                x_min -= k * abs(x_max - x_min)
                y_min -= k * abs(y_max - y_min)
                x_max += k * abs(x_max - x_min)
                y_max += k * abs(y_max - y_min)

        if is_flip:
            # Random flip
            if np.random.random_sample() <= 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                x_max_2 = width - x_min
                x_min_2 = width - x_max
                x_max = x_max_2
                x_min = x_min_2
                gaze_x = 1 - gaze_x

        if is_color_distortion:
            # Random color change
            if np.random.random_sample() <= 0.5:
                img = TF.adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

        head_channel = imutils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
                                                    resolution=self.scene_size, coordconv=False).unsqueeze(0)

        # exam face box
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width, x_max)
        y_max = min(height, y_max)
        
        x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
        face_center =  torch.Tensor([x_center / width , y_center / height])
        # Crop the face
        face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        if self.imshow:
            img.save("img_aug.jpg")
            face.save('face_aug.jpg')

        if self.scene_transform is not None:
            img = self.scene_transform(img)
        if self.face_transform is not None:
            face = self.face_transform(face)
            
        # generate head position zp
        head_position_zp = i2c(face_center.unsqueeze(0)).squeeze(0)[2]  # 先升维成batch的形式，再降维成向量
        head_position = i2c(face_center.unsqueeze(0)).squeeze(0)
        # input q_i and p_i, generate gaze direction 
        gaze_ds = ds(face_center.unsqueeze(0), PoG.unsqueeze(0)).squeeze(0)

        # generate the heat map used for deconv prediction
        # gaze_heatmap = torch.zeros([height, width])  # set the size of the output
        # gaze_heatmap = imutils.draw_labelmap(gaze_heatmap, [gaze_x * width, gaze_y * height], 11, type='Gaussian')
        
        if self.imshow:
            # torchvision.utils.save_image(gaze_heatmap, 'gaze_heatmap.jpg',
			# 						 normalize=True, scale_each=True, range=(0, 1), nrow=1)
            torchvision.utils.save_image(head_channel, 'head_channel.jpg',
									 normalize=True, scale_each=True, range=(0, 1), nrow=1)
            fig = plt.figure(111)
            img = imutils.unnorm(img.numpy())
            img = np.clip(img, 0, 1)
            plt.imshow(np.transpose(img, (1, 2, 0)))
            # plt.imshow(np.array(Image.fromarray(gaze_heatmap.numpy()).resize(self.scene_size)), cmap='jet', alpha=0.3)
            plt.imshow(np.array(Image.fromarray(1 - head_channel.squeeze(0).numpy()).resize(self.scene_size)), alpha=0.2)
            plt.savefig('viz_aug.png')
    
        # gaze_heatmap 真实图的大小, Tensor的大小不一致，不能用dataloader返回
        # face: (3, 224, 224) type is tensor 
        # head_position_zp: head position zp
        # gaze_ds: ds的真实值
        # imsize: 原图的尺寸
        # face_center: 归一化的人脸中心值(0, 1)
        # scene_path
        # PoG: 归一化的Point of Gaze
    
        img_imf = {}
        img_imf["size"] = imsize
        img_imf["face_center_raw"] = torch.Tensor([x_center, y_center])
        img_imf["face_center_norm"] = face_center
        img_imf["scene_path"] = scene_path
        img_imf["pog_norm"] = PoG
        img_imf["wh"] = x_max - x_min

        if self.test:
            return face, torch.Tensor(head_position), gaze_ds, img_imf, PoG
        else:
            return face, torch.Tensor(head_position), gaze_ds, img_imf
        
        # if self.test:
        #     return face, torch.Tensor([head_position_zp]), gaze_ds, img_imf, PoG
        # else:
        #     return face, torch.Tensor([head_position_zp]), gaze_ds, img_imf
    
    
    def __len__(self):
        return self.metadata['gaze'].shape[0]
        # return 100

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
    # 输入image坐标系中的两点，输出subject坐标系中的视线方向
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

if __name__ == "__main__":
    # Data path
    data_test_path = "/home/data/tbw_gaze/training_dataset/gazefollow360/GazeFollow360_dataset"
    testdataset = GazeFollow360(data_test_path, split='test', imshow=True)

    for i in range(0, len(testdataset)):
        face, zp, gaze_ds, img_imf, pog = testdataset[i]
        print(face, zp, gaze_ds, img_imf, pog)
        break
