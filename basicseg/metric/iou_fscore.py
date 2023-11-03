import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class Binary_metric():
    'calculate fscore and iou'
    def __init__(self, thr=0.5):
        self.mean_reset()
        self.norm_reset()
        self.thr = thr
        self.cnt = 0

    def mean_reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def norm_reset(self):
        self.norm_metric = {'precision':0., 'recall':0., 'fscore':0., 'iou':0.}
        self.cnt = 0.

    def update(self, pred, target):
        # for safety
        pred = pred.detach().clone()
        target = target.detach().clone()
        if torch.max(pred) > 1.1:
            pred = torch.sigmoid(pred)
        pred[pred >= self.thr] = 1
        pred[pred < self.thr] = 0
        self.cur_tp = torch.sum((pred == target) * target, dim=(1,2,3))
        self.cur_tn = torch.sum((pred == target) * (1 - target), dim=(1,2,3))
        self.cur_fp = torch.sum((pred != target) * pred, dim=(1,2,3))
        self.cur_fn = torch.sum((pred != target) * (1 - pred), dim=(1,2,3))
        self.tp += self.cur_tp.sum()
        self.tn += self.cur_tn.sum()
        self.fp += self.cur_fp.sum()
        self.fn += self.cur_fn.sum()
        norm_result = self.norm_compute()
        for k in self.norm_metric.keys():
            self.norm_metric[k] += norm_result[k]
        self.cnt += pred.shape[0]

    def get_mean_result(self):
        mean_metric = self.mean_compute()
        self.mean_reset()
        return mean_metric

    def get_norm_result(self):
        for k,v in self.norm_metric.items():
            self.norm_metric[k] /= self.cnt
        norm_metric = deepcopy(self.norm_metric)
        self.norm_reset()
        return norm_metric

    def norm_compute(self):
        eps = 1e-6
        precision = (self.cur_tp / (self.cur_tp + self.cur_fp + eps)).sum()
        recall = (self.cur_tp / (self.cur_tp + self.cur_fn + eps)).sum()
        fscore = (2 * precision * recall / (precision + recall + eps)).sum()
        iou = (self.cur_tp / (self.cur_tp + self.cur_fn + self.cur_fp + eps)).sum()
        return {"precision":precision, "recall":recall, "fscore":fscore, "iou":iou}

    def mean_compute(self):
        eps = 1e-6
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        fscore = 2 * precision * recall / (precision + recall + eps)
        iou = self.tp / (self.tp + self.fn + self.fp + eps)
        return {"precision":precision, "recall":recall, "fscore":fscore, "iou":iou}