import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data as Data
import os
import time
import logging
import random
import copy
import numpy as np
from basicseg.seg_model import Seg_model
from basicseg.utils.yaml_options import parse_options, dict2str
from basicseg.utils.path_utils import *
from basicseg.utils.logger import get_root_logger, init_tb_logger, get_env_info, MessageLogger
from basicseg.data import build_dataset

def set_seed(seed, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if cuda_deterministic:
        # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # faster
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def init_exp(opt, args):
    exp_name = opt['exp'].get('name')
    if not exp_name:
        exp_name = os.path.basename(args.opt[:-4])
        opt['exp']['name'] = exp_name
    exp_root = make_exp_root(os.path.join('experiment', exp_name))
    opt['exp']['exp_root'] = exp_root
    log_file = os.path.join(exp_root, f'train_{exp_name}_{get_time_str()}.log')
    logger = get_root_logger(logger_name='basicseg', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    tb_logger = init_tb_logger(log_dir = os.path.join(exp_root, 'tb_log'))
    return logger, tb_logger

def init_model(opt):

    model = Seg_model(opt)
    return model

def init_dataset(opt):
    # trainset
    train_opt = opt['dataset']['train']
    trainset = build_dataset(train_opt)
    test_opt = opt['dataset']['test']
    testset = build_dataset(test_opt)
    return trainset, testset

def init_dataloader(opt, trainset, testset):
    if opt['exp']['dist']:
        sampler = Data.DistributedSampler(trainset)
    else:
        sampler = None
    train_loader = Data.DataLoader(dataset=trainset, batch_size=opt['exp']['bs'],\
                                    sampler=sampler, num_workers=opt['exp'].get('nw', 16))
    test_loader  = Data.DataLoader(dataset=testset, batch_size=opt['exp']['bs'],\
                                    sampler=None, num_workers=opt['exp'].get('nw', 16))
    return train_loader, test_loader

def main():
    opt, args = parse_options()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt['exp']['device']) # not safe there
    if isinstance(opt['exp']['device'], int):
        opt['exp']['dist'] = False
        cur_rank = 0
        total_device = 1
        opt['exp']['num_devices'] = total_device
    elif isinstance(opt['exp']['device'], str):
        opt['exp']['dist'] = True
        dist.init_process_group(backend='nccl')
        total_device = len(opt['exp']['device']) // 2 + 1
        opt['exp']['num_devices'] = total_device
        cur_rank = dist.get_rank()

    # init dataset
    trainset, testset = init_dataset(opt)
    train_loader, test_loader = init_dataloader(opt, trainset, testset)

    # init exp_root, logger, tb_logger
    total_epochs = opt['exp']['total_epochs']
    total_iters = total_epochs * (len(trainset) // opt['exp']['bs'] // total_device +1)
    opt['exp']['total_iters'] = total_iters
    save_interval = opt['exp']['save_interval']
    test_interval = opt['exp']['test_interval']
    logger, tb_logger = init_exp(opt, args)
    set_seed(cur_rank + 0)
    # 初始化 模型参数, 包含 网络 优化器 损失函数 学习率准则
    # initialize parameters including network, optimizer, loss function, learning rate scheduler
    model = init_model(opt)
    cur_iter = 0
    cur_epoch = 1
    # 从断点继续训练
    # train from checkpoint
    if opt.get('resume'):
        if opt['resume'].get('net_path'):
            model.load_network(model.net, opt['resume']['net_path'])
            logger.info(f'load pretrained network from: {opt["resume"]["net_path"]}')
        else:
            logger.info(f'load from random initialized network')
        if opt['resume'].get('state_path'):
            cur_epoch = model.resume_training(opt['resume']['state_path'])
            cur_iter = cur_epoch * (len(trainset) // opt['exp']['bs'] // total_device + 1)
            logger.info(f'resume training from epoch: {cur_epoch}')
        else:
            logger.info(f'training from epoch: 1')

    msg_logger = MessageLogger(opt, start_epoch=cur_epoch, tb_logger=tb_logger)
    for epoch in range(cur_epoch, total_epochs+1):
        if opt['exp']['dist']:
            train_loader.sampler.set_epoch(epoch)
        epoch_st_time = time.time()
        ########## training ##########
        for idx, data in enumerate(train_loader):
            cur_iter += 1
            model.update_learning_rate(cur_iter, idx)
            model.optimize_one_iter(data)
        epoch_time = time.time() - epoch_st_time
        log_vars = {'epoch': epoch}
        log_vars.update({'lrs': model.get_current_learning_rate()})
        log_vars.update({'time': epoch_time})
        log_vars.update({'train_loss': model.get_epoch_loss(opt['exp']['dist'], 'sum')})
        log_vars.update({'train_mean_metric': model.get_mean_metric(opt['exp']['dist'], 'mean')})
        log_vars.update({'train_norm_metric': model.get_norm_metric(opt['exp']['dist'], 'mean')})
        ########## tesing ##########
        if cur_rank == 0 and epoch % test_interval == 0:
            # model.net.eval()
            model.model_to_eval()
            for idx, data in enumerate(test_loader):
                model.test_one_iter(data)
            log_vars.update({'test_loss': model.get_epoch_loss()})
            test_mean_metric = model.get_mean_metric()
            test_norm_metric = model.get_norm_metric()
            log_vars.update({'test_mean_metric': test_mean_metric})
            log_vars.update({'test_norm_metric': test_norm_metric})
            if test_mean_metric['iou'] > model.best_mean_metric['iou']:
                model.best_mean_metric['iou'] = test_mean_metric['iou']
                model.best_mean_metric['net'] = copy.deepcopy(model.net.state_dict())
                model.best_mean_metric['epoch'] = epoch
            if test_norm_metric['iou'] > model.best_norm_metric['iou']:
                model.best_norm_metric['iou'] = test_norm_metric['iou']
                model.best_norm_metric['net'] = copy.deepcopy(model.net.state_dict())
                model.best_norm_metric['epoch'] = epoch
            # model.net.train()
            model.model_to_train()
        ########## saving_model ##########
        if cur_rank == 0 and epoch % save_interval == 0 :
            model.save_network(opt, model.net, epoch)
            model.save_training_state(opt, epoch)

        msg_logger(log_vars)

    ########## trainging done ##########
    if cur_rank == 0:
        model.save_network(opt, model.net, current_epoch='latest')
        model.save_network(opt, model.best_mean_metric['net'], current_epoch='best_mean', net_dict=True)
        model.save_network(opt, model.best_norm_metric['net'], current_epoch='best_norm', net_dict=True)
        logger.info(f"best_mean_metric: [epoch: {model.best_mean_metric['epoch']}] [iou: {model.best_mean_metric['iou']:.4f}]")
        logger.info(f"best_norm_metric: [epoch: {model.best_norm_metric['epoch']}] [iou: {model.best_norm_metric['iou']:.4f}]")

if __name__ == '__main__':
    main()