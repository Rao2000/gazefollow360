import os
import random
import torch
import torch.nn as nn

from model import gde_dp, gazefollow360
from dataset import GazeFollow360
from config import *
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import argparse
from datetime import datetime
import shutil
import logging 
import numpy as np
import cv2
from utils import imutils, evaluation
from PIL import Image
from model import i2c
from tensorboardX import SummaryWriter
import warnings
import torchsnooper
from tqdm import tqdm
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_logger(output_path):
    logger = logging.getLogger()
    logging.basicConfig(filename=os.path.join(output_path, 'run_log.log'),
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger

def get_gt(img_imf_batch):
    output_size = [56, 56]
    gaze_heatmap_list = []
    for PoG in img_imf_batch["pog_norm"]:
        gaze_x, gaze_y = PoG[0], PoG[1]
        gaze_heatmap = torch.zeros(output_size)
        gaze_heatmap = imutils.draw_labelmap(gaze_heatmap, [gaze_x * output_size[0], gaze_y * output_size[1]], 3, type="Gaussian")
        gaze_heatmap = cv2.resize(np.array(gaze_heatmap), (56, 56))
        gaze_heatmap_list.append(gaze_heatmap)
    return torch.Tensor(gaze_heatmap_list)

def func(PoG, imsize, output):
    cos_loss1 = nn.CosineSimilarity(dim=1, eps=1e-6)

    gaze = PoG.numpy()
    shape = imsize.numpy().astype(np.int64)
    # AUC: area under curve of ROC
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

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="gpu id")
parser.add_argument("--init_weights", type=str, default='/home/data/tbw_gaze/gaze323/gaze323-gaze-gazefollow360-/model/all/epoch_10_weights.pt', help="initial weights")
parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=512, help="batch size")
parser.add_argument("--epochs", type=int, default=25, help="number of epochs")
parser.add_argument("--print_every", type=int, default=10, help="print every ___ iterations")
parser.add_argument("--eval_every", type=int, default=500, help="evaluate every ___ iterations")
parser.add_argument("--test_every", type=int, default=5, help="test every ___ epochs")
parser.add_argument("--save_every", type=int, default=1, help="save every ___ epochs")
parser.add_argument("--log_dir", type=str, default="/home/data/tbw_gaze/gaze323/log_2023/all/", help="directory to save log files")
parser.add_argument("--logdir_name", type=str, default="all17", help="subdir to save log files")
args = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

setup_seed(1037)

def train():
    # Prepare data
    print("Loading Data")
    train_dataset = GazeFollow360(dataset_root_path, split="train")
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               drop_last=True,
                                               shuffle=True,
                                               num_workers=16)

    # val_dataset = GazeFollow360(dataset_root_path, split="vali")
    # val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
    #                                            batch_size=args.batch_size,
    #                                            shuffle=False,
    #                                            num_workers=16)
    
    test_dataset = GazeFollow360(dataset_root_path, split="test")
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=16)

    # Set up log dir
    logdir = os.path.join(args.log_dir, args.logdir_name)
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)

    logger = get_logger(logdir)
    writer = SummaryWriter(logdir)
    np.random.seed(1037)

    # Define device
    device = torch.device('cuda', args.device)

    # Load model
    logger.info("Constructing model")
    model = gazefollow360()
    model.cuda()
    
    # if args.init_weights:
    #     model_dict = model.state_dict()
    #     pretrained_dict = torch.load(args.init_weights, map_location=torch.device('cpu'))
    #     pretrained_dict = pretrained_dict['model']
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict)

    # Loss functions
    cos_loss = nn.CosineSimilarity(dim=1, eps=1e-6)
    mse_loss = nn.MSELoss() # not reducing in order to ignore outside cases
    bcelogit_loss = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    step = 0
    loss_amp_factor = 100 # multiplied to the loss to prevent underflow
    max_steps = len(train_loader)
    optimizer.zero_grad()

    logger.info("Training in progress ...")
    for ep in range(args.epochs):
        for batch, (face, head_position, gaze_ds, img_imf) in enumerate(train_loader):

            # 字典内的
            model.train()
            face = face.cuda()
            head_position = head_position.cuda()

            output = model(face, head_position, img_imf)

            heatmap_gt = get_gt(img_imf)
            heatmap_gt = heatmap_gt.cuda()

            output = output.reshape(-1, 56, 56)
            
            loss = mse_loss(heatmap_gt, output)

            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if batch % args.print_every == 0:
                logger.info("Epoch:{:04d}\tstep:{:06d}/{:06d}\ttraining loss :{:.4f}".format(ep, batch+1, max_steps, loss))

                writer.add_scalar("Train_Loss", loss, global_step=step)

            # if (batch != 0 and batch % args.eval_every == 0) or batch+1 == max_steps:
            #     logger.info('Validation in progress ...')
            #     model.train(False)
            #     AUC = []; pixel_dist = []; norm_dist = []
            #     with torch.no_grad():
            #         for val_idx, (val_face, val_head_position_zp, val_gaze_ds, img_imf) in enumerate(val_loader):
            #             output_resolution = [224, 224]
            #             val_face = val_face.cuda()
            #             val_head_position_zp = val_head_position_zp.cuda()
            #             output = model(val_face, val_head_position_zp, img_imf)
            #             heatmap_gt = get_gt(img_imf)
            #             heatmap_gt = heatmap_gt.reshape(-1, 224, 224).cuda()
            #             output = output.reshape(-1, 224, 224)
            #             gt_gaze = img_imf["pog_norm"]
            #             imsize = img_imf["size"]
            
            #             all_distances = []
            #             for b_i in range(heatmap_gt.shape[0]): 
            #                 gaze = gt_gaze[b_i].numpy()
            #                 shape = imsize[b_i].numpy()
                            
            #                 pred_x, pred_y = evaluation.argmax_pts(output[b_i].cpu().numpy())
            #                 norm_p = [pred_x/float(output_resolution[1]), pred_y/float(output_resolution[0])]
                            
            #                 distances = evaluation.L2_dist(gaze, norm_p, shape)
            #                 all_distances.append(distances)
            #                 norm_distance = evaluation.L2_dist(gaze, norm_p, [1, 1])
            #                 norm_dist.append(norm_distance)
            #             pixel_dist.append(torch.mean(torch.tensor(all_distances)))
            #     logger.info("\tmin dist:{:.4f}\tnorm dist:{:.4f}".format(torch.mean(torch.tensor(pixel_dist)), torch.mean(torch.tensor(norm_dist))))
            #     writer.add_scalar('Validation_min_dist', torch.mean(torch.tensor(pixel_dist)), global_step=step)     

        if (ep + 1) % args.test_every == 0:
            logger.info('Test in progress ...')
            model.eval()
            AUC = []; pixel_dist = []; norm_dist = []; sphere_dist = []; loss_list = []
            with torch.no_grad():
                for test_idx, (test_face, test_head_position, test_gaze_ds, img_imf, PoG) in tqdm(enumerate(test_loader)):

                    test_face = test_face.cuda()
                    test_head_position = test_head_position.cuda()

                    output = model(test_face, test_head_position, img_imf)

                    heatmap_gt = get_gt(img_imf)
                    heatmap_gt = heatmap_gt.cuda()

                    output = output.reshape(-1, 56, 56)
                    imsize = img_imf["size"]

                    test_loss = mse_loss(heatmap_gt, output)

                    loss_list.append(test_loss)

                    def Callback(data):
                        AUC.append(data[0])
                        pixel_dist.append(data[1])
                        norm_dist.append(data[2])
                        sphere_dist.append(data[3])
        
                    def err_call_back(err):
                        print(f'3出错啦~ error：{str(err)}')

                    ctx = torch.multiprocessing.get_context("spawn")
                    pool = ctx.Pool(16)
                    for PoG, imsize, output in zip(PoG, imsize, output.cpu()):
                        pool.apply_async(func, (PoG, imsize, output), callback=Callback, error_callback=err_call_back)
                    pool.close()
                    pool.join()

            pixel_dist = torch.mean(torch.tensor(pixel_dist))
            norm_dist = torch.mean(torch.tensor(norm_dist))
            auc = torch.mean(torch.tensor(AUC))
            sphere_dist = torch.mean(torch.tensor(sphere_dist))
            test_loss = torch.mean(torch.tensor(loss_list))

            logger.info("Test dataset\tpixel dist:{:.4f}\tnorm dist:{:.4f}\tAUC:{:.4f}\tsphere dist:{:.4f}\tloss:{:.4f}".format(pixel_dist, norm_dist, auc, sphere_dist, test_loss))
            writer.add_scalar('Test_pixel_dist', pixel_dist, global_step=step)
            writer.add_scalar('Test_norm_dist', norm_dist, global_step=step)
            writer.add_scalar('Test_AUC', auc, global_step=step)
            writer.add_scalar('Test_sphere_dist', sphere_dist, global_step=step)
            writer.add_scalar('Test_loss', test_loss, global_step=step)
            
        if ep % args.save_every == 0:
            # save the model
            checkpoint = {'model': model.state_dict()}
            torch.save(checkpoint, os.path.join(logdir, 'epoch_%02d_weights.pt' % (ep+1)))

def eval():
    test_dataset = GazeFollow360(dataset_root_path, split="test")
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=8)
    # Define device
    device = torch.device('cuda', args.device)
    model = gazefollow360()
    model.cuda().to(device)
    model_path = "/home/data/tbw_gaze/gaze323/log_2023/all/all14/epoch_05_weights.pt"
    state_dict = torch.load(model_path, map_location=device)
    print("> Loaded model parameters from: %s" % model_path)
    print("> {}".format(model.load_state_dict(state_dict["model"])))
    model.train(False)
    pd_data = []
    gaze_heatmap_pred_list = []
    with torch.no_grad():
        for test_idx, (test_face, test_head_position_zp, test_gaze_ds, img_imf, PoG) in tqdm(enumerate(test_loader)):
            test_face = test_face.cuda().to(device)
            test_head_position_zp = test_head_position_zp.cuda().to(device)

            output = model(test_face, test_head_position_zp, img_imf)

            output = output.reshape(-1, 64, 64)
            gaze_heatmap_pred = output.cpu().numpy()
            gaze_heatmap_pred_list.append(gaze_heatmap_pred)
            
            imsize = img_imf["size"]
            picname = img_imf["scene_path"]
            face_center_raw = img_imf["face_center_raw"]
            imsize = img_imf["size"]
            
            for b_i in range(PoG.shape[0]):
                print(PoG.shape[0]) 
                gaze = PoG[b_i].numpy()
                shape = imsize[b_i].numpy()
                scene_path = picname[b_i]
                face_center = face_center_raw[b_i].numpy()
                
                pred_x, pred_y = evaluation.argmax_pts(output[b_i].cpu().numpy())
                norm_p = [pred_x / 56, pred_y / 56]
                
                data = {"picname": scene_path, "center_x": face_center[0], "center_y": face_center[1],
                        "predicted_gaze_x": norm_p[0], "predicted_gaze_y": norm_p[1],
                        "gaze_label_x":gaze[0], "gaze_label_y":gaze[1], "imsize_x":shape[0], "imsize_y":shape[1]}
                pd_data.append(data)
    
    pd_data = pd.DataFrame(pd_data)
    logdir = "/home/data/tbw_gaze/gaze323/gaze323-gaze-gazefollow360-/result/qym"
    pd_data.to_csv(os.path.join(logdir, "predicts.csv"), sep='\t')
    gaze_heatmap = np.concatenate(gaze_heatmap_pred_list, axis=0)
    np.save(os.path.join(logdir, "predict_heatmaps.npy"), gaze_heatmap)

if __name__ == "__main__":
    train()
    # eval()

