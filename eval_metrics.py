import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm

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

def L2_distance(pred_x, pred_y, gt_x, gt_y):
    # 以batch形式输入
    # 返回欧氏距离batch
    assert pred_x.shape[0] == pred_y.shape[0] == gt_x.shape[0] == gt_y.shape[0]
    return np.sqrt(np.square(pred_x - gt_x) + np.square(pred_y - gt_y))

def angle_error(a, b):
    # 以batch形式输入
    # 返回弧度batch
    assert a.shape[0] == b.shape[0] and a.shape[1] == b.shape[1] == 3
    multi = np.sum(np.multiply(a, b), axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return np.arccos(np.true_divide(multi, np.multiply(norm_a, norm_b)))

def multi_hot_targets(gaze_pts, out_res):
    w, h = out_res
    target_map = np.zeros((h, w))
    gaze_pts = [gaze_pts]
    for p in gaze_pts:
        if p[0] >= 0:
            x, y = map(int,[p[0]*w, p[1]*h])
            x = min(x, w-1)
            y = min(y, h-1)
            target_map[y, x] = 1
    return target_map

def pixel_distance(pred_x, pred_y, gt_x, gt_y, imsize_x, imsize_y):
    # 以batch形式输入
    # 返回平均像素距离
    assert pred_x.shape[0] == pred_y.shape[0] == gt_x.shape[0] == gt_y.shape[0] == imsize_x.shape[0] == imsize_y.shape[0]
    row_pred_x = np.multiply(pred_x, imsize_x)
    row_pred_y = np.multiply(pred_y, imsize_y)
    row_gt_x = np.multiply(gt_x, imsize_x)
    row_gt_y = np.multiply(gt_y, imsize_y)
    return np.mean(L2_distance(row_pred_x, row_pred_y, row_gt_x, row_gt_y))

def norm_distance(pred_x, pred_y, gt_x, gt_y):
    # 以batch形式输入
    # 返回平均归一化距离
    assert pred_x.shape[0] == pred_y.shape[0] == gt_x.shape[0] == gt_y.shape[0]
    return np.mean(L2_distance(pred_x, pred_y, gt_x, gt_y))

def sphere_distance(pred_x, pred_y, gt_x, gt_y):
    # 以batch形式输入
    # 返回平均弧度误差
    assert pred_x.shape[0] == pred_y.shape[0] == gt_x.shape[0] == gt_y.shape[0]
    pred_point_I = np.array(list(zip(pred_x, pred_y)))
    gt_point_I = np.array(list(zip(gt_x, gt_y)))
    pred_point_C = I2C(pred_point_I)
    gt_point_C = I2C(gt_point_I)
    angle_pred_gt = angle_error(pred_point_C, gt_point_C)
    return np.mean(angle_pred_gt)

def auc(pred_heatmap, gt_x, gt_y, imsize_x, imsize_y, is_img=True):
    # 非batch输入
    gt_heatmap = multi_hot_targets((gt_x, gt_y), (imsize_x, imsize_y))
    scaled_heatmap = np.array(Image.fromarray(pred_heatmap).resize((imsize_x, imsize_y)))
    if is_img:
        auc_score = roc_auc_score(np.reshape(gt_heatmap, gt_heatmap.size), np.reshape(scaled_heatmap, scaled_heatmap.size))
    else:
        auc_score = roc_auc_score(gt_heatmap, scaled_heatmap)
    return auc_score

def eval_metrics(information_path, pred_heatmap_path):
    imformation_dataframe = pd.read_csv(information_path, sep='\t')

    pred_x = np.array(imformation_dataframe.loc[:, 'predicted_gaze_x'])
    pred_y = np.array(imformation_dataframe.loc[:, 'predicted_gaze_y'])
    gt_x = np.array(imformation_dataframe.loc[:, 'gaze_label_x'])
    gt_y = np.array(imformation_dataframe.loc[:, 'gaze_label_y'])
    imsize_x = np.array(imformation_dataframe.loc[:, 'imsize_x'])
    imsize_y = np.array(imformation_dataframe.loc[:, 'imsize_y'])
    pred_heatmap = np.load(pred_heatmap_path)

    norm_dist = norm_distance(pred_x, pred_y, gt_x, gt_y)
    pixel_dist = pixel_distance(pred_x, pred_y, gt_x, gt_y, imsize_x, imsize_y)
    sphere_dist = sphere_distance(pred_x, pred_y, gt_x, gt_y)

    auc_list = []
    for idx in tqdm(range(0, pred_heatmap.shape[0])):
        # 只能输入单通道热图
        # auc_list.append(auc(pred_heatmap[idx][0], gt_x[idx], gt_y[idx], imsize_x[idx], imsize_y[idx]))
        auc_list.append(auc(pred_heatmap[idx], gt_x[idx], gt_y[idx], int(imsize_x[idx]), int(imsize_y[idx])))
    auc_score = np.mean(np.array(auc_list))
    # auc_score = 0
    print("exp: {}\tnorm dist: {}\tpixel dist: {}\tsphere dist: {}\tauc: {}".format(information_path.split('/')[-2], norm_dist, pixel_dist, sphere_dist, auc_score))

if __name__ == "__main__":
    # results_root_dir = '/home/data/tbw_gaze/gaze323/gaze323-gaze-GazeFollowing-/result'
    # sub_dirs = os.listdir(results_root_dir)
    # for sub_dir in sub_dirs:
    #     result_root = os.path.join(results_root_dir, sub_dir)
    #     information_path = os.path.join(result_root, 'predicts.csv')
    #     pred_heatmap_path = os.path.join(result_root, 'predict_heatmaps.npz')
    #     eval_metrics(information_path, pred_heatmap_path)

    results_root_dir = '/home/data/tbw_gaze/gaze323/gaze323-gaze-gazefollow360-/result/gazefollow360'
    information_path = os.path.join(results_root_dir, 'predicts.csv')
    pred_heatmap_path = os.path.join(results_root_dir, 'predict_heatmaps.npy')
    eval_metrics(information_path, pred_heatmap_path)

    # x1 = np.array([0.5, 1])
    # y1 = np.array([1, 0])
    # x2 = np.array([0.2, 0])
    # y2 = np.array([0.5, 1])
    # print(L2_distance(x1, y1, x2, y2))
