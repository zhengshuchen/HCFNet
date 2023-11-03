import torch
import torch.nn as nn
from basicseg.base_model import Base_model
import copy
from collections import OrderedDict
from basicseg.metric import Binary_metric
import torch.nn.functional as F

class Test_model(Base_model):
    def __init__(self, opt):
        self.opt = opt
        self.init_model()

    def init_model(self):
        self.setup_net()
        self.metric = Binary_metric()

    def get_mean_metric(self, dist=False, reduction='mean'):
        if dist:
            return self.reduce_dict(self.metric.get_mean_result(), reduction)
        else:
            return self.dict_wrapper(self.metric.get_mean_result())

    def get_norm_metric(self, dist=False, reduction='mean'):
        if dist:
            return self.reduce_dict(self.metric.get_norm_result(), reduction)
        else:
            return self.dict_wrapper(self.metric.get_norm_result())
    def test_one_iter(self, data):
        with torch.no_grad():
            img, mask = data
            img, mask = img.to(self.device), mask.to(self.device)
            pred = self.net(img)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            pred = F.interpolate(pred, mask.shape[2:], mode='bilinear', align_corners=False)
            self.metric.update(pred=pred, target=mask)
        return pred
    def infer_one_iter(self, data):
        with torch.no_grad():
            img = data
            img = img.to(self.device)
            pred = self.net(img)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            # for loss_type, loss_fn in self.loss_fn.items():
            #     loss = loss_fn(pred, mask)
            #     self.epoch_loss[loss_type] += loss.detach().clone()
            #     self.batch_loss[loss_type] = loss.detach().clone()
            pred = F.interpolate(pred, img.shape[2:], mode='bilinear', align_corners=False)
            # self.metric.update(pred=pred, target=mask)
        return pred