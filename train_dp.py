import os
import random
import torch
import torch.nn as nn

from model import GDE_network, DP_network, gde_dp
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
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from utils import imutils, evaluation
from PIL import Image
from tensorboardX import SummaryWriter
import warnings
import torchsnooper
from torchvision import transforms
from tqdm import tqdm

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

def get_gt_heatmap(img_imf_batch, distant_xyxy_batch, local_xyxy_batch):
    heatmap_gt1_list, heatmap_gt2_list = [], []
    output_size = [224, 224]  # w, h
    for imsize, PoG, distant_xyxy, local_xyxy in zip(img_imf_batch["size"], img_imf_batch["pog_norm"], distant_xyxy_batch, local_xyxy_batch):
        width, height = imsize
        gaze_x, gaze_y = PoG[0], PoG[1]
        gaze_heatmap = torch.zeros(output_size)
        gaze_heatmap = imutils.draw_labelmap(gaze_heatmap, [gaze_x * output_size[0], gaze_y * output_size[1]], 5, type="Gaussian")

        heatmap_gt1 = gaze_heatmap[int(distant_xyxy[1]*output_size[1]):int(distant_xyxy[3]*output_size[1]), int(distant_xyxy[0]*output_size[0]):int(distant_xyxy[2]*output_size[0])]
        heatmap_gt2 = gaze_heatmap[int(local_xyxy[1]*output_size[1]):int(local_xyxy[3]*output_size[1]), int(local_xyxy[0]*output_size[0]):int(local_xyxy[2]*output_size[0])]
        heatmap_gt1 = cv2.resize(np.array(heatmap_gt1), (56, 56))
        heatmap_gt2 = cv2.resize(np.array(heatmap_gt2), (56, 56))
        heatmap_gt1_list.append(heatmap_gt1)
        heatmap_gt2_list.append(heatmap_gt2)
    return torch.Tensor(heatmap_gt1_list), torch.Tensor(heatmap_gt2_list)

class Lion(torch.optim.Optimizer):
  r"""Implements Lion algorithm."""

  def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
    """Initialize the hyperparameters.
    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
      lr (float, optional): learning rate (default: 1e-4)
      betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.99))
      weight_decay (float, optional): weight decay coefficient (default: 0)
    """

    if not 0.0 <= lr:
      raise ValueError('Invalid learning rate: {}'.format(lr))
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
    defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
    super().__init__(params, defaults)

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.
    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    Returns:
      the loss.
    """
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue

        # Perform stepweight decay
        p.data.mul_(1 - group['lr'] * group['weight_decay'])

        grad = p.grad
        state = self.state[p]
        # State initialization
        if len(state) == 0:
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(p)

        exp_avg = state['exp_avg']
        beta1, beta2 = group['betas']

        # Weight update
        update = exp_avg * beta1 + grad * (1 - beta1)
        p.add_(torch.sign(update), alpha=-group['lr'])
        # Decay the momentum running average coefficient
        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

    return loss

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="gpu id")
parser.add_argument("--init_weights", type=str, default='/home/data/tbw_gaze/gaze323/log_2023/dp/dp16/epoch_25_weights.pt', help="initial weights")
parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=320, help="batch size")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--print_every", type=int, default=10, help="print every ___ iterations")
parser.add_argument("--eval_every", type=int, default=200, help="evaluate every ___ iterations")
parser.add_argument("--test_every", type=int, default=5, help="evaluate every ___ epochs")
parser.add_argument("--save_every", type=int, default=1, help="save every ___ epochs")
parser.add_argument("--log_dir", type=str, default="/home/data/tbw_gaze/gaze323/log_2023/dp", help="directory to save log files")
parser.add_argument("--logdir_name", type=str, default="dp19", help="subdir to save log files")
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
    model = gde_dp()
    model.cuda()
    
    # if args.init_weights:
    #     model_dict = model.state_dict()
    #     pretrained_dict = torch.load(args.init_weights)
    #     pretrained_dict = pretrained_dict['model']
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict)

    # Loss functions
    cos_loss = nn.CosineSimilarity(dim=1, eps=1e-6)
    mse_loss = nn.MSELoss() # not reducing in order to ignore outside cases
    bce_loss = nn.BCELoss()
    # bcelogit_loss = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = Lion(model.parameters(), lr=args.lr)

    step = 0
    loss_amp_factor = 10000 # multiplied to the loss to prevent underflow
    max_steps = len(train_loader)
    optimizer.zero_grad()

    logger.info("Training in progress ...")
    for ep in range(args.epochs):
        train_loss1_list = []
        train_loss2_list = []
        for batch, (face, head_position, gaze_ds, img_imf) in enumerate(train_loader):
            # 字典内的
            
            model.train()
            face = face.cuda()
            head_position = head_position.cuda()

            output1, output2, distant_xyxy, local_xyxy = model(face, head_position, img_imf)

            heatmap_gt1, heatmap_gt2 = get_gt_heatmap(img_imf, distant_xyxy, local_xyxy)
            heatmap_gt1 = heatmap_gt1.cuda()
            heatmap_gt2 = heatmap_gt2.cuda()

            output1 = output1.reshape(-1, 56, 56)
            output2 = output2.reshape(-1, 56, 56)
            
            loss1 = mse_loss(heatmap_gt1, output1) 
            loss2 = mse_loss(heatmap_gt2, output2)

            train_loss1_list.append(loss1)
            train_loss2_list.append(loss2)

            loss1.backward()
            loss2.backward()

            optimizer.step()
            optimizer.zero_grad()
            
            step += 1

            if batch % args.print_every == 0:
                logger.info("Epoch:{:04d}\tstep:{:06d}/{:06d}\ttraining loss 1 :{:.4f}".format(ep, batch+1, max_steps, loss1))
                logger.info("Epoch:{:04d}\tstep:{:06d}/{:06d}\ttraining loss 2 :{:.4f}".format(ep, batch+1, max_steps, loss2))

        train_loss1 = torch.mean(torch.tensor(train_loss1_list))
        train_loss2 = torch.mean(torch.tensor(train_loss2_list))
        writer.add_scalar("Train_Loss1", train_loss1, global_step=step)
        writer.add_scalar("Train_Loss2", train_loss2, global_step=step)
        
        #     if (batch != 0 and batch % args.eval_every == 0) or batch+1 == max_steps:
        #         logger.info('Validation in progress ...')
        #         model.train(False)
        #         AUC = []
        #         with torch.no_grad():
        #             for val_idx, (val_face, val_head_position_zp, val_gaze_ds, img_imf) in enumerate(val_loader):
        #                 val_face = val_face.cuda()
        #                 val_head_position_zp = val_head_position_zp.cuda()
        #                 output1, output2, distant_xyxy, local_xyxy = model(face, head_position_zp, img_imf)
            
        #                 heatmap_gt1, heatmap_gt2 = get_gt_heatmap(img_imf, distant_xyxy, local_xyxy)
                        
        #                 for b_i in range(heatmap_gt1.shape[0]): 
        #                     pred1 = np.array(output1[b_i].cpu())
        #                     gt1 = np.array(heatmap_gt1[b_i]) 
                            
        #                     # AUC: area under curve of ROC
        #                     auc_score1 = evaluation.auc(pred1, gt1)
        #                     AUC.append(auc_score1)
        #                     pred2 = np.array(output2[b_i])
        #                     gt2 = np.array(heatmap_gt2[b_i]) 
                            
        #                     # AUC: area under curve of ROC
        #                     auc_score2 = evaluation.auc(pred2, gt2)
        #                     AUC.append(auc_score2)
        #         logger.info("val dataset AUC:{:.4f}".format(torch.mean(torch.tensor(AUC))))
        #         writer.add_scalar('Validation_AUC', torch.mean(torch.tensor(AUC)), global_step=step)
                            
        if (ep+1) % args.test_every == 0:
            logger.info('Test in progress ...')
            model.eval()
            test_loss1_list = []
            test_loss2_list = []
            test_norm_list = []
            with torch.no_grad():
                for test_idx, (test_face, test_head_position, test_gaze_ds, img_imf, PoG) in tqdm(enumerate(test_loader)):
                    test_face = test_face.cuda()
                    test_head_position = test_head_position.cuda()
                    output1, output2, distant_xyxy, local_xyxy = model(test_face, test_head_position, img_imf)

                    output1 = output1.reshape(-1, 56, 56)
                    output2 = output2.reshape(-1, 56, 56)
            
                    heatmap_gt1, heatmap_gt2 = get_gt_heatmap(img_imf, distant_xyxy, local_xyxy)
                    heatmap_gt1 = heatmap_gt1.cuda()
                    heatmap_gt2 = heatmap_gt2.cuda()

                    test_loss1 = mse_loss(heatmap_gt1, output1) 
                    test_loss2 = mse_loss(heatmap_gt2, output2)
                    test_loss1_list.append(test_loss1)
                    test_loss2_list.append(test_loss2)
            
            test_loss1 = torch.mean(torch.tensor(test_loss1_list))
            test_loss2 = torch.mean(torch.tensor(test_loss2_list))

            logger.info("test loss1:{:.4f}".format(test_loss1))
            logger.info("test loss2:{:.4f}".format(test_loss2))
            
            writer.add_scalar('test_loss1', test_loss1, global_step=step)
            writer.add_scalar('test_loss2', test_loss2, global_step=step)

        if ep % args.save_every == 0:
            # save the model
            checkpoint = {'model': model.state_dict()}
            torch.save(checkpoint, os.path.join(logdir, 'epoch_%02d_weights.pt' % (ep+1)))

if __name__ == "__main__":
    train()
