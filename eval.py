import torch
from torchvision import transforms
import torch.nn as nn

from model import gazefollow360
from dataset import GazeFollow360
from config import *
from utils import imutils, evaluation
from PIL import Image

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import numpy as np
import warnings
from train_all import get_gt
from tqdm import *
from model import i2c
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

logdir = "/home/data"
if not os.path.exists(logdir):
    os.makedirs(logdir)
    
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="gpu id")
parser.add_argument("--model_weights", type=str, default="/home/data/tbw_gaze/gaze323/log_2023/all/all14/epoch_05_weights.pt", help="model weights")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
args = parser.parse_args()

def test():
    # Prepare data
    print("Loading Data")
    test_dataset = GazeFollow360(dataset_root_path, split="test")
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=16) 

    # Define device
    device = torch.device('cuda', args.device)

    # Load model
    print("Constructing model")

    # Define pd data
    pd_data = []

    model = gazefollow360()
    model.cuda()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.model_weights)
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print('Evaluation in progress ...')
    model.eval()
    AUC = []; pixel_dist = []; norm_dist = []; sphere_dist = []
    with torch.no_grad():
        gaze_heatmap_pred_list = []
        for test_idx, (test_face, test_head_position_zp, test_gaze_ds, img_imf, PoG) in tqdm(enumerate(test_loader)):

            test_face = test_face.cuda().to(device)
            test_head_position_zp = test_head_position_zp.cuda().to(device)
            val_gaze_heatmap_pred = model(test_face, test_head_position_zp, img_imf)
            
            val_gaze_heatmap_pred = val_gaze_heatmap_pred.reshape(-1, 56, 56)
            val_gaze_heatmap_pred = val_gaze_heatmap_pred.cpu().numpy()
            gaze_heatmap_pred_list.append(val_gaze_heatmap_pred)
            
            imsize = img_imf["size"]
            path = img_imf["scene_path"]
            for b_i in range(PoG.shape[0]):
                gaze = PoG[b_i].numpy().tolist()
                scene_path = path[b_i]
                shape = imsize[b_i].numpy()
                pred_x, pred_y = evaluation.argmax_pts(val_gaze_heatmap_pred[b_i])
                norm_p = [pred_x/float(val_gaze_heatmap_pred[b_i].shape[0]), pred_y/float(val_gaze_heatmap_pred[b_i].shape[1])]
                data = {"picname": scene_path, "predicted_gaze_x": norm_p[0], "predicted_gaze_y": norm_p[1], 
                        "gaze_label_x": gaze[0], "gaze_label_y": gaze[1],
                        "imsize_x": shape[0], "imsize_y": shape[1]}
                pd_data.append(data)
    pd_data = pd.DataFrame(pd_data)
    pd_data.to_csv(os.path.join(logdir, "predicts.csv"), sep='\t')
    gaze_heatmap_pred = np.concatenate(gaze_heatmap_pred_list, axis=0)
    np.save(os.path.join(logdir, "predict_heatmaps.npy"), gaze_heatmap_pred)    

if __name__ == "__main__":
    test()
