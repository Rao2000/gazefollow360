from __future__ import absolute_import
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def auc(heatmap, onehot_im, is_im=True):
    if is_im:
        auc_score = roc_auc_score(np.reshape(onehot_im,onehot_im.size), np.reshape(heatmap,heatmap.size))
    else:
        auc_score = roc_auc_score(onehot_im, heatmap)
    return auc_score

def auc1(heatmap, onehot_im, is_im=True):
    auc_score = roc_auc_score(onehot_im, heatmap)
    return auc_score


def ap(label, pred):
    return average_precision_score(label, pred)


def argmax_pts(heatmap):
    idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    pred_y, pred_x = map(float,idx)
    return pred_x, pred_y


def L2_dist(p1, p2, imsize):
    return np.sqrt((p1[0] * imsize[0] - p2[0] * imsize[0])**2 + (p1[1] * imsize[1] - p2[1] * imsize[1])**2)

def pixel_error(a, b, img_shape):
    assert(a.ndim == b.ndim)
    assert(a.ndim > 1)
    a = np.multiply(a, img_shape)
    b = np.multiply(b, img_shape)
    squared_difference = np.square(a - b)
    ssd = np.sum(squared_difference, axis=tuple(range(1, a.ndim)))
    return np.sqrt(ssd)