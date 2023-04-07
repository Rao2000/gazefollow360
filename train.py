import os
import random
import torch
import torch.nn as nn

# from model import GDE_network, DP_network
from model import *
from dataset import GazeFollow360
from config import *
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3'
import argparse
from datetime import datetime
import shutil
import logging 
import numpy as np
from tensorboardX import SummaryWriter
from torchvision import transforms
import warnings
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
parser.add_argument("--init_weights", type=str, default='/home/data/tbw_gaze/gaze323/log_2023/gde/gde23/epoch_20_weights.pt', help="initial weights")
parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--print_every", type=int, default=10, help="print every ___ iterations")
parser.add_argument("--eval_every", type=int, default=200, help="evaluate every ___ iterations")
parser.add_argument("--test_every", type=int, default=1, help="evaluate every ___ epochs")
parser.add_argument("--save_every", type=int, default=1, help="save every ___ epochs")
parser.add_argument("--log_dir", type=str, default="/home/data/tbw_gaze/gaze323/log_2023/gde", help="directory to save log files")
parser.add_argument("--logdir_name", type=str, default="gde24", help="subdir to save log files")
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

num = 423
setup_seed(num)

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
    np.random.seed(num)

    # Load model
    logger.info("Constructing model")
    # model = GDE_network(resnet_feature_dim=64, fc_feature_dim=16)
    model = GDE_network()
    model.cuda()
    
    # weight = torch.load("/home/data/tbw_gaze/gaze323/gaze323-gaze-gazefollow360-/RepVGG-B3-200epochs-train.pth")
    # model.repvgg_b3.load_state_dict(weight)
    
    # if args.init_weights:
    #     model_dict = model.state_dict()
    #     # pretrained_dict = torch.load(args.init_weights, map_location=torch.device('cpu'))
    #     pretrained_dict = torch.load(args.init_weights)
    #     pretrained_dict = pretrained_dict['model']
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict)

    # Loss functions
    cos_loss = nn.CosineSimilarity(dim=1, eps=1e-6)
    mse_loss = nn.MSELoss(reduce=False) # not reducing in order to ignore outside cases
    # bcelogit_loss = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = Lion(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)

    step = 0
    loss_amp_factor = 10000 # multiplied to the loss to prevent underflow
    max_steps = len(train_loader)
    optimizer.zero_grad()

    logger.info("Training in progress ...")
    angle_min = 200
    epoch_min = 0
    for ep in range(args.epochs):
        for batch, (face, head_position, gaze_ds, img_imf) in enumerate(train_loader):
            # 字典内的
            model.train()
            face = face.cuda()
            head_position = head_position.cuda()
            gaze_ds_predict = model(face, head_position)
            gaze_ds = gaze_ds.cuda()

            # gaze_ds_s2c = torch.tensor(s2c(gaze_ds.cpu())).cuda()

            # loss1 = 1 - torch.mean(cos_loss(gaze_ds, gaze_ds_predict))
            # loss1 = torch.arccos(torch.mean(cos_loss(gaze_ds_s2c, gaze_ds_predict)))
            loss1 = torch.arccos(torch.mean(cos_loss(gaze_ds, gaze_ds_predict)))
            angle = loss1 / np.pi * 180
            loss1.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if batch % args.print_every == 0:
                logger.info("Epoch:{:04d}\tstep:{:06d}/{:06d}\ttraining loss:{:.4f}\tangle:{:.2f}".format(ep, batch+1, max_steps, loss1, angle))
                # Tensorboard
                # ind = np.random.choice(len(images), replace=False)
                writer.add_scalar("Train_Loss", loss1, global_step=step)

            # if (batch != 0 and batch % args.eval_every == 0) or batch+1 == max_steps:
            #     logger.info('Validation in progress ...')
            #     model.eval()
            #     cos_err = []
            #     with torch.no_grad():
            #         for val_idx, (val_face, val_head_position_zp, val_gaze_ds, img_imf) in enumerate(val_loader):
            #             val_face = val_face.cuda()
            #             val_head_position_zp = val_head_position_zp.cuda()
            #             gaze_ds_predict = model(val_face, val_head_position_zp)
            #             val_gaze_ds = val_gaze_ds.cuda()

            #             # val_gaze_ds_s2c = torch.tensor(s2c(val_gaze_ds.cpu())).cuda()
                        
            #             # cos_sim = torch.mean(F.cosine_similarity(val_gaze_ds, gaze_ds_predict, dim=1))
            #             cos_sim = torch.mean(F.cosine_similarity(val_gaze_ds, gaze_ds_predict, dim=1))
                        
            #             cos_err.append(cos_sim.item())

            #     ds_valid = np.arccos(np.mean(cos_err))
            #     angle_valid = ds_valid / np.pi * 180
                        
            #     # logger.info("gaze direction ds is {}".format(np.mean(cos_err)))
            #     logger.info("gaze direction ds is {}\tangle:{:.2f}".format(ds_valid, angle_valid))

            #     # Tensorboard
            #     # writer.add_scalar('Validation AUC', torch.mean(torch.tensor(AUC)), global_step=step)
            #     # writer.add_scalar('Validation_cos_sim', np.mean(cos_err), global_step=step)
            #     writer.add_scalar('Validation_cos_sim', ds_valid, global_step=step)

        if (ep + 1) % args.test_every == 0:
            logger.info('Test in progress ...')
            model.eval()
            cos_err = []
            sphere_dist_list = []
            with torch.no_grad():
                for test_idx, (test_face, test_head_position, test_gaze_ds, img_imf, PoG) in enumerate(test_loader):
                    test_face = test_face.cuda()
                    test_head_position = test_head_position.cuda()
                    gaze_ds_predict = model(test_face, test_head_position)
                    test_gaze_ds = test_gaze_ds.cuda()

                    head = i2c(img_imf["face_center_norm"])

                    gaze_pred_c = gaze_point(head, gaze_ds_predict.cpu().numpy())
                    gaze_gt_c = i2c(PoG.numpy())
                    sphere_dist = torch.mean(cos_loss(torch.tensor(gaze_gt_c), torch.tensor(gaze_pred_c)))
                    sphere_dist_list.append(sphere_dist.item())

                    cos_sim = torch.mean(cos_loss(test_gaze_ds, gaze_ds_predict))
                    cos_err.append(cos_sim.item())  

            ds_test = np.arccos(np.mean(cos_err))
            angle_test = ds_test / np.pi * 180

            sphere_test = np.arccos(np.mean(sphere_dist_list))
            
            if angle_test < angle_min:
               angle_min = angle_test
               epoch_min = ep
                        
            # logger.info("gaze direction ds is {}".format(np.mean(cos_err)))
            logger.info("gaze direction sphere dist is {}\tangle:{:.2f}\tangle_min:{:.2f}\tepoch_min:{}".format(ds_test, angle_test, angle_min, epoch_min))
            logger.info("final is {:.4f}".format(sphere_test))
            # Tensorboard
            # writer.add_scalar('Validation AUC', torch.mean(torch.tensor(AUC)), global_step=step)
            # writer.add_scalar('Test_min_dist', np.mean(cos_err), global_step=ep)
            writer.add_scalar('Test_sphere_dist', ds_test, global_step=ep)
            writer.add_scalar('Test_angle_error', angle_test, global_step=ep)
            writer.add_scalar('Test_final', sphere_test, global_step=ep)

        if ep % args.save_every == 0:
            # save the model
            checkpoint = {'model': model.state_dict()}
            torch.save(checkpoint, os.path.join(logdir, 'epoch_%02d_weights.pt' % (ep+1)))

if __name__ == "__main__":
    train()
