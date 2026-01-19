import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

import os
import shutil
# import logging
import cv2
import numpy as np
from PIL import Image

import sys
import time
import pandas as pd
import pdb
from torchvision import transforms

def metric_s_for_null(pred):
    num_seg, T, H, W = pred.shape
    pred = pred.view(num_seg*T, H, W)
    assert len(pred.shape) == 3

    N = pred.size(0)
    num_pixels = pred.view(-1).shape[0] 

    temp_pred = torch.sigmoid(pred)
    pred = (temp_pred > 0.5).int()

    x = torch.sum(pred.view(-1)) 
    s = torch.sqrt(x / num_pixels)

    return s

def mask_iou(pred, target, eps=1e-7, size_average=True):
    r"""
        param:
            pred: size [N x H x W]
            target: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    # return mask_iou_224(pred, target, eps=1e-7)
    num_ref, T, H, W = pred.shape
    pred = pred.view(num_ref*T, H, W)
    target = target.view(num_ref*T, H, W)
    assert len(pred.shape) == 3 and pred.shape == target.shape

    N = pred.size(0)
    # 像素数
    num_pixels = pred.size(-1) * pred.size(-2)
    # gt是否是纯黑
    no_obj_flag = (target.sum(2).sum(1) == 0)

    # 会把pred进行sigmoid变为01之间的概率
    temp_pred = torch.sigmoid(pred)
    # 通过阈值变成01矩阵
    pred = (temp_pred > 0.4).int()
    # 交集
    inter = (pred * target).sum(2).sum(1)
    # 并集
    union = torch.max(pred, target).sum(2).sum(1)

    inter_no_obj = ((1 - target) * (1 - pred)).sum(2).sum(1)
    inter[no_obj_flag] = inter_no_obj[no_obj_flag]
    union[no_obj_flag] = num_pixels

    iou = torch.sum(inter / (union + eps)) / N

    return iou.item()


def _eval_pr(y_pred, y, num, device='cuda'):
    if device.startswith('cuda'):
        prec, recall = torch.zeros(num).to(y_pred.device), torch.zeros(num).to(y_pred.device)
        # 0到1 生成num个阈值
        thlist = torch.linspace(0, 1 - 1e-10, num).to(y_pred.device)
    else:
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)

    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()

        # 计算 True Positives（TP）
        tp = (y_temp * y).sum()
        # 一个是交集除以 预测面积 一个是除以真实面积
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)

    return prec, recall


def Eval_Fmeasure(pred, gt, measure_path, pr_num=255, device='cuda'):
    r"""
        param:
            pred: size [N x H x W]
            gt: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    num_ref, T, H, W = pred.shape
    pred = pred.view(num_ref*T, H, W)
    gt = gt.view(num_ref*T, H, W)
    assert len(pred.shape) == 3


    # sigmoid转为01之间的
    pred = torch.sigmoid(pred) 
    N = pred.size(0)
    beta2 = 0.3
    avg_f, img_num = 0.0, 0
    score = torch.zeros(pr_num)


    for img_id in range(N):
        if torch.mean(gt[img_id]) == 0.0:
            continue
        prec, recall = _eval_pr(pred[img_id], gt[img_id], pr_num, device=device)
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
        f_score[f_score != f_score] = 0  # for Nan
        avg_f += f_score
        img_num += 1
        score = avg_f / img_num

    return score.max().item()

