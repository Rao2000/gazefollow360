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
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="gpu id")
parser.add_argument("--model_weights", type=str, default="/home/data/tbw_gaze/gaze323/log_2023/all/all14/epoch_05_weights.pt", help="model weights")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
args = parser.parse_args()

def func(PoG, imsize, output):

    cos_loss1 = nn.CosineSimilarity(dim=1, eps=1e-6)

    gaze = PoG.numpy()
    shape = imsize.numpy().astype(np.int64)
    # AUC: area under curve of ROC
    multi_hot = torch.zeros(output_resolution, output_resolution)
    multi_hot = imutils.draw_labelmap(multi_hot, [gaze * output_resolution, gaze * output_resolution], 3, type='Gaussian')
    multi_hot = (multi_hot > 0).float() * 1 # make GT heatmap as binary labels
    multi_hot = misc.to_numpy(multi_hot)
    multi_hot = imutils.multi_hot_targets(gaze, shape)
    scaled_heatmap = np.array(Image.fromarray(output.cpu().numpy()).resize(shape.tolist()))
    auc_score = evaluation.auc(scaled_heatmap, multi_hot)
    
    # px distance
    pred_x, pred_y = evaluation.argmax_pts(output.cpu().numpy())
    norm_p = [pred_x / 56, pred_y / 56]

    norm_p_cemara = torch.tensor(i2c(np.array([norm_p])))
    gaze_cemara = torch.tensor(i2c(np.array([gaze])))
    sphere_distance = torch.arccos(cos_loss1(gaze_cemara, norm_p_cemara))
    
    distances = evaluation.L2_dist(gaze, norm_p, shape)
    norm_distance = evaluation.L2_dist(gaze, norm_p, [1, 1])
    return [auc_score, distances, norm_distance, sphere_distance]

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
        with torch.no_grad():
            for test_idx, (test_face, test_head_position_zp, test_gaze_ds, img_imf, PoG) in tqdm(enumerate(test_loader)):

                test_face = test_face.cuda().to(device)
                test_head_position_zp = test_head_position_zp.cuda().to(device)
                output = model(test_face, test_head_position_zp, img_imf)

                output = output.reshape(-1, 56, 56)

                imsize = img_imf["size"]

                def Callback(data):
                    AUC.append(data[0])
                    pixel_dist.append(data[1])
                    norm_dist.append(data[2])
                    sphere_dist.append(data[3])
    
                def err_call_back(err):
                    print(f'3出错啦~ error：{str(err)}')

                ctx = torch.multiprocessing.get_context("spawn")
                pool = ctx.Pool(16)
                # for b_i in range(PoG.shape[0]):
                for PoG, imsize, output in zip(PoG, imsize, output.cpu()):
                    pool.apply_async(func, (PoG, imsize, output), callback=Callback, error_callback=err_call_back)
                pool.close()
                pool.join()

        print("\tAUC:{:.4f}\tpixel dist:{:.4f}\tnorm dist:{:.4f}\tsphere dist:{:.4f}".format(
              torch.mean(torch.tensor(AUC)),
              torch.mean(torch.tensor(pixel_dist)),
              torch.mean(torch.tensor(norm_dist)),
              torch.mean(torch.tensor(sphere_dist))))

if __name__ == "__main__":
    test()
