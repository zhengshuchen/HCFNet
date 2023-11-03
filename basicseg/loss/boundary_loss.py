import numpy as np
from scipy.ndimage import distance_transform_edt as eucl_distance
from basicseg.utils.registry import LOSS_REGISTRY
import torch.nn as nn
import torch
'''
boundary loss at: https://arxiv.org/abs/1812.07032
modified from: https://github.com/LIVIAETS/boundary-loss only for binary classification
'''
def get_dist_map(mask):
    # 注:mask  numpy:array [h,w]
    # res = np.zeros_like(mask, dtype=np.float32)
    mask = mask.clone().detach()
    posmask = mask.numpy().astype(np.bool_)
    resolution = [1, 1]
    negmask = ~posmask
    res = eucl_distance(negmask, sampling=resolution) * negmask - (
            eucl_distance(posmask, sampling=resolution) - 1) * posmask
    res = np.clip(res, a_min=0, a_max=None)
    return res

@LOSS_REGISTRY.register()
class BD_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(BD_loss, self).__init__()
        self.reduction = reduction
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        bd_loss = pred * target
        if self.reduction == 'mean':
            return bd_loss.mean()
        elif self.reduction == 'sum':
            return bd_loss.sum()
