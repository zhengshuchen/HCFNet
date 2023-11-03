import torch
import torch.nn as nn
import torch.nn.functional as F
from basicseg.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class Iou_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.eps = 1e-6
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        iou = intersection_sum / (pred_sum + target_sum - intersection_sum + self.eps)
        if self.reduction == 'mean':
            return 1 - iou.mean()
        elif self.reduction == 'sum':
            return 1 - iou.mean()
        else: 
            raise NotImplementedError('reduction type {} not implemented'.format(self.reduction))

@LOSS_REGISTRY.register()
class Dice_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.eps = 1e-6
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = torch.sum(pred * target, dim=(1,2,3))
        total_sum = torch.sum((pred + target), dim=(1,2,3))
        dice = 2 * intersection / (total_sum + self.eps)
        if self.reduction == 'mean':
            return 1 - dice.mean()
        elif self.reduction == 'sum':
            return 1 - dice.sum()
        else: 
            raise NotImplementedError('reduction type {} not implemented'.format(self.reduction))

@LOSS_REGISTRY.register()
class Bce_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.eps = 1e-6
    def forward(self, pred, target):
        loss_fn = nn.BCEWithLogitsLoss(reduction=self.reduction)
        return loss_fn(pred, target)

@LOSS_REGISTRY.register()
class L1_loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.eps = 1e-6
    def forward(self, pred, target):
        loss_fn = nn.L1Loss(reduction=self.reduction)
        return loss_fn(pred, target)