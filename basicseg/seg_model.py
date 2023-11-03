import torch
import torch.nn as nn
from basicseg.base_model import Base_model
import copy
from collections import OrderedDict
from basicseg.metric import Binary_metric
import torch.nn.functional as F

class Seg_model(Base_model):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.init_model()

    def init_model(self):
        self.setup_net()
        self.setup_optimizer()
        self.setup_loss()
        self.setup_metric()
        self.setup_lr_schduler()

    def setup_metric(self):
        self.best_norm_metric = {'epoch':0., 'iou':0., 'net': None}
        self.best_mean_metric = {'epoch':0, 'iou':0., 'net': None}
        self.metric = Binary_metric()
        self.epoch_metric = {}
        self.batch_metric = {}
        
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

    def get_epoch_loss(self, dist=False, reduction='sum'):
        epoch_loss = copy.deepcopy(self.epoch_loss)
        self.reset_epoch_loss()
        if dist:
            return self.reduce_dict(epoch_loss, reduction)
        else:
            return self.dict_wrapper(epoch_loss)

    def get_batch_loss(self, dist=False, reduction='sum'):
        batch_loss = copy.deepcopy(self.batch_loss)
        self.reset_batch_loss()
        if dist:
            return self.reduce_dict(batch_loss, reduction)
        else:
            return self.dict_wrapper(batch_loss)

    def optimize_one_iter(self, data):
        if self.bd_loss:
            img, mask, dist_map = data
            img, mask, dist_map = img.to(self.device), mask.to(self.device), dist_map.to(self.device)
        else:
            img, mask = data
            img, mask = img.to(self.device), mask.to(self.device)
        pred, pred_1, pred_2,pred_3,pred_4 = self.net(img)
        cur_loss = 0.
        if not isinstance(pred, (list, tuple)):
            pred = [pred]
        for idx, pred_ in enumerate(pred):
            pred_ = F.interpolate(pred_, mask.shape[2:], mode='bilinear', align_corners=False)
            if idx == 0:
                pred[0] = pred_
            for loss_type, loss_criteria in self.loss_fn.items():
                if loss_type == 'BD_loss':
                    loss = loss_criteria(pred_, dist_map) * self.loss_weight[loss_type][idx]
                else:
                    loss = loss_criteria(pred_, mask) * self.loss_weight[loss_type][idx] +loss_criteria(pred_1, mask) * 0.5 +loss_criteria(pred_2, mask) * 0.25+loss_criteria(pred_3, mask) * 0.125+loss_criteria(pred_4, mask) * 0.0625
                self.epoch_loss[loss_type + '_' + str(idx)] += loss.detach().clone()
                self.batch_loss[loss_type + '_' + str(idx)] += loss.detach().clone()
                cur_loss += loss
        self.optim.zero_grad()
        cur_loss.backward()
        self.optim.step()
        with torch.no_grad():
            self.metric.update(pred=pred[0], target=mask)
        # return loss_result

    def test_one_iter(self, data):
        with torch.no_grad():
            if self.bd_loss:
                img, mask, dist_map = data
                img, mask, dist_map = img.to(self.device), mask.to(self.device), dist_map.to(self.device)
            else:
                img, mask = data
                img, mask = img.to(self.device), mask.to(self.device)
            pred,_,_,_,_ = self.net(img)
            if not isinstance(pred, (list, tuple)):
                pred = [pred]
            for idx, pred_ in enumerate(pred):
                pred_ = F.interpolate(pred_, mask.shape[2:], mode='bilinear', align_corners=False)
                if idx == 0:
                    pred[0] = pred_
                for loss_type, loss_criteria in self.loss_fn.items():
                    if loss_type == 'BD_loss':
                        loss = loss_criteria(pred_, dist_map) * self.loss_weight[loss_type][idx]
                    else:
                        loss = loss_criteria(pred_, mask) * self.loss_weight[loss_type][idx]
                    self.epoch_loss[loss_type + '_' + str(idx)] += loss.detach().clone()
                    self.batch_loss[loss_type + '_' + str(idx)] += loss.detach().clone()
            self.metric.update(pred=pred[0], target=mask)

