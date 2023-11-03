# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import datetime
import logging
import time

from .dist_util import get_dist_info, master_only


class MessageLogger():
    """Message logger for printing.

    Args:
        opt (dict): Config. It contains the following keys:
            name (str): Exp name.
            logger (dict): Contains 'print_freq' (str) for logger interval.
            train (dict): Contains 'total_iter' (int) for total iters.
            use_tb_logger (bool): Use tensorboard logger.
        start_iter (int): Start iter. Default: 1.
        tb_logger (obj:`tb_logger`): Tensorboard logger. Default： None.
    """

    def __init__(self, opt, start_epoch=1, tb_logger=None):
        self.exp_name = opt['exp']['name']
        self.interval = opt['exp']['log_interval']
        self.start_epoch = start_epoch
        self.max_epochs = opt['exp']['total_epochs']
        self.use_tb_logger = True
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()

    @master_only
    def __call__(self, log_vars):
        """Format logging message.
        Args:
            log_vars (dict): It contains the following keys:
                epoch (int): Epoch number.
                iter (int): Current iter.
                lrs (list): List for learning rates.

                time (float): Iter time.
                data_time (float): Data time for each iter.
        """
        # epoch, iter, learning rates
        current_epoch = log_vars.pop('epoch')
        # current_iter = log_vars.pop('iter')
        # total_iter = log_vars.pop('total_iter')
        lrs = log_vars.pop('lrs')

        message = (f'[{self.exp_name}][epoch:{current_epoch:3d}, '
                   f'lr:(')
        message += f'{lrs:.3e},'
        message += ')] '

        # time and estimated time
        if 'time' in log_vars.keys():
            epoch_time = log_vars.pop('time')
            total_time = time.time() - self.start_time
            time_sec_avg = total_time / (current_epoch - self.start_epoch + 1)
            eta_sec = time_sec_avg * (self.max_epochs - current_epoch)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            message += f'[eta: {eta_str}, '
            message += f'time (epoch): {epoch_time:.3f} ] '

        # other items, especially losses
        for k, v in log_vars.items():
            # message += f'{k}: {v:.4e} '
            # tensorboard logger
            if self.use_tb_logger and 'debug' not in self.exp_name:
                # normed_step = 10000 * (current_iter / total_iter)
                # normed_step = int(normed_step)

                if k == 'train_loss':
                    message += '\nTrainSet\n'
                    for loss_type, loss_value in log_vars[k].items():
                        self.tb_logger.add_scalar(f'train_losses/{loss_type}', loss_value, current_epoch)
                        message += f'{loss_type}:{loss_value:.4e}  '
                    message += '\n'
                elif k == 'train_mean_metric':
                    for metric_type, metric_value in log_vars[k].items():
                        self.tb_logger.add_scalar(f'train_mean_metrics/{metric_type}', metric_value, current_epoch)
                    message += f"m_fscore:{log_vars[k]['fscore']:.4f}  m_iou:{log_vars[k]['iou']:.4f}  "
                elif k == 'train_norm_metric':
                    for metric_type, metric_value in log_vars[k].items():
                        self.tb_logger.add_scalar(f'train_norm_metrics/{metric_type}', metric_value, current_epoch)
                    message += f"n_fscore:{log_vars[k]['fscore']:.4f}  n_iou:{log_vars[k]['iou']:.4f}  "
                elif k == 'test_loss':
                    message += '\nTestSet\n'
                    for loss_type, loss_value in log_vars[k].items():
                        self.tb_logger.add_scalar(f'test_losses/{loss_type}', loss_value, current_epoch)
                        message += f'{loss_type}:{loss_value:.4e}  '
                    message += '\n'
                elif k == 'test_mean_metric':
                    for metric_type, metric_value in log_vars[k].items():
                        self.tb_logger.add_scalar(f'test_mean_metrics/{metric_type}', metric_value, current_epoch)
                    message += f"m_fscore:{log_vars[k]['fscore']:.4f}  m_iou:{log_vars[k]['iou']:.4f}  "
                elif k == 'test_norm_metric':
                    for metric_type, metric_value in log_vars[k].items():
                        self.tb_logger.add_scalar(f'test_norm_metrics/{metric_type}', metric_value, current_epoch)
                    message += f"n_fscore:{log_vars[k]['fscore']:.4f}  n_iou:{log_vars[k]['iou']:.4f}  "
                else:
                    assert 1 == 0
                # else:
                #     self.tb_logger.add_scalar(k, v, current_iter)
        message += '\n'
        self.logger.info(message)


@master_only
def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger





def get_root_logger(logger_name='basicseg',
                    log_level=logging.INFO,
                    log_file=None):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added.

    Args:
        logger_name (str): root logger name. Default: 'basicsr'.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(format=format_str, level=log_level)
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger


def get_env_info():
    """Get environment information.

    Currently, only log the software version.
    """
    import torch
    import torchvision
    msg = ('\nVersion Information: '
            f'\n\tPyTorch: {torch.__version__}'
            f'\n\tTorchVision: {torchvision.__version__}')
    return msg

# @master_only
# def init_wandb_logger(opt):
#     """We now only use wandb to sync tensorboard log."""
#     import wandb
#     logger = logging.getLogger('basicsr')
#
#     project = opt['logger']['wandb']['project']
#     resume_id = opt['logger']['wandb'].get('resume_id')
#     if resume_id:
#         wandb_id = resume_id
#         resume = 'allow'
#         logger.warning(f'Resume wandb logger with id={wandb_id}.')
#     else:
#         wandb_id = wandb.util.generate_id()
#         resume = 'never'
#
#     wandb.init(
#         id=wandb_id,
#         resume=resume,
#         name=opt['name'],
#         config=opt,
#         project=project,
#         sync_tensorboard=True)
#
#     logger.info(f'Use wandb logger with id={wandb_id}; project={project}.')