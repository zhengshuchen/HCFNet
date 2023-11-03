import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from basicseg.networks import build_network
from basicseg.loss import build_loss
import basicseg.utils.lr_scheduler as lr_scheduler
from basicseg.utils.dist_util import get_dist_info
import copy
import os
from basicseg.utils.dist_util import master_only
from torch.nn.parallel import DistributedDataParallel
from collections import OrderedDict
from basicseg.utils.path_utils import make_dir
logger = logging.getLogger('basicseg')

class Base_model():
    def __init__(self):
        pass
    def model_to_train(self):
        self.net.train()
    def model_to_eval(self):
        self.net.eval()
    def setup_net(self):
        self.net = build_network(self.opt['model']['net'])
        self.rank, _ = get_dist_info()
        self.device = torch.device(self.rank)
        if self.opt['exp'].get('dist', False):
            self.total_rank = self.opt['exp']['num_devices']
            self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net).to(self.device)
            self.net = DistributedDataParallel(self.net, device_ids=[self.rank], output_device=self.rank)
        else:
            self.net.to(self.device)


    def setup_optimizer(self):
        optim_type = self.opt['model']['optim'].pop('type')
        # optim_params = [param for param in self.net.parameters() if param.requires_grad]
        self.init_lr = self.opt['model']['optim'].pop('init_lr')
        if optim_type == 'AdamW':
            self.optim = torch.optim.AdamW(self.net.parameters(), self.init_lr, **self.opt['model']['optim'])
        elif optim_type == 'SGD':
            self.optim = torch.optim.SGD(self.net.parameters(), self.init_lr, **self.opt['model']['optim'])
        else:
            raise NotImplementedError('optim type {} not implemented yet'.format(optim_type))

    def forward_once(self):
        self.net.eval()
        base_dim = self.opt['model']['net'].get('in_c', 3)
        with torch.no_grad():
            h = self.opt['dataset']['train']['img_sz']
            ipt = torch.rand(1,base_dim,h,h).to(self.device)
            pred,_,_,_,_ = self.net(ipt)
            if isinstance(pred, (tuple, list)):
                pred_nums = len(pred)
            else:
                pred_nums = 1
            del ipt
            del pred
        self.net.train()
        return pred_nums

    def setup_loss(self):
        loss_opt = self.opt['model']['loss']
        self.loss_fn = {}
        self.loss_weight = {}
        self.epoch_loss = {}
        self.batch_loss = {}
        self.bd_loss = False
        if self.opt['dataset']['train'].get('bd_loss', False):
            self.bd_loss = True
        for loss in loss_opt:
            self.loss_fn[loss_opt[loss]['type']] =\
                build_loss(loss_opt[loss])
            self.loss_weight[loss_opt[loss]['type']] = loss_opt[loss]['weight']
            pred_nums = self.forward_once()
            if not isinstance(self.loss_weight[loss_opt[loss]['type']], (tuple, list)):
                self.loss_weight[loss_opt[loss]['type']] = [self.loss_weight[loss_opt[loss]['type']]] * pred_nums
            assert len(self.loss_weight[loss_opt[loss]['type']]) == pred_nums
            for idx in range(pred_nums):
                self.epoch_loss[loss_opt[loss]['type'] + '_' + str(idx)] = 0.
                self.batch_loss[loss_opt[loss]['type'] + '_' + str(idx)] = 0.
        assert len(self.loss_fn) > 0

    def reset_epoch_loss(self):
        for k in self.epoch_loss.keys():
            self.epoch_loss[k] = 0.

    def reset_batch_loss(self):
        for k in self.batch_loss.keys():
            self.batch_loss[k] = 0.

    def setup_lr_schduler(self):
        lr_opt = self.opt['model']['lr']
        self.step_interval = lr_opt['scheduler'].pop('step_interval', 'epoch')
        if self.step_interval == 'epoch':
            self.T_max = self.opt['exp']['total_epochs']
        elif self.step_interval == 'iter':
            self.T_max = self.opt['exp']['total_iters']

        self.warmup_iter = lr_opt['warmup_iter']
        scheduler_type = lr_opt['scheduler'].pop('type')
        if scheduler_type is None:
            self.scheduler = None
        else:
            if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
                    self.scheduler = \
                        lr_scheduler.MultiStepRestartLR(self.optim,
                                                        **lr_opt['scheduler'])
            elif scheduler_type == 'CosineAnnealingRestartLR':
                    self.scheduler = \
                        lr_scheduler.CosineAnnealingRestartLR(
                            self.optim, **lr_opt['scheduler'])
            elif scheduler_type == 'CosineAnnealingLR':
                self.scheduler = \
                    torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, self.T_max, **lr_opt['scheduler'])
            elif scheduler_type == 'Poly':
                self.scheduler = \
                    lr_scheduler.PolyLR(self.optim, self.T_max, **lr_opt['scheduler'])
            else:
                raise NotImplementedError(
                    f'Scheduler {scheduler_type} is not implemented yet.')
    def update_learning_rate(self, current_iter, idx):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
                Default： -1.
        """
        if current_iter > 1:
            if self.scheduler:
                if self.step_interval == 'iter':
                    self.scheduler.step()
                elif self.step_interval == 'epoch':
                    if idx == 0:
                        self.scheduler.step()
                else:
                    return NotImplementedError(f'scheduler step_interval {self.step_interval} error')
        # set up warm-up learning rate
        if current_iter < self.warmup_iter:
            # get initial lr for each group
            # modify warming-up learning rates
            # currently only support linearly warm up
            warmup_lr = \
                self.init_lr / self.warmup_iter * current_iter
            # set learning rate
            for param in self.optim.param_groups:
                param['lr'] = warmup_lr

    def get_current_learning_rate(self):
        return self.optim.param_groups[0]['lr']

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DistributedDataParallel)):
            net = net.module
        return net

    @master_only
    def save_network(self, opt, net, current_epoch, param_key='params', net_label='net', net_dict=False):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        save_filename = f'{net_label}_{current_epoch}.pth'
        save_path = os.path.join(opt['exp']['exp_root'], 'models', save_filename)
        make_dir(os.path.dirname(save_path))
        if net_dict:
            state_dict = OrderedDict()
            for key, param in net.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    state_dict[key[7:]] = param.cpu()
                else:
                    state_dict[key] = param.cpu()
            torch.save(state_dict, save_path)

        else:
            net_ = self.get_bare_model(net)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                # actually this report error, but it won't be used so doesn't occur
                # if key.startswith('module.'):  # remove unnecessary 'module.'
                #     key = key[7:]
                state_dict[key] = param.cpu()
            torch.save(state_dict, save_path)

    def load_network(self, net, load_path, strict=True, param_key=None, **kwargs):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        net = self.get_bare_model(net)
        logger.info(
            f'Loading {net.__class__.__name__} model from {load_path}.')
        load_net = torch.load(
            load_path, map_location='cpu')
        if param_key is not None:
            load_net = load_net[param_key]
        # print(' load net keys', load_net.keys)
        # remove unnecessary 'module.'
        for k, v in copy.deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    @master_only
    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print keys with differnet name or different size when loading models.

        1. Print keys with differnet names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            logger.warning('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f'  {v}')
            logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(
                        f'Size different, ignore [{k}]: crt_net: '
                        f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    @master_only
    def save_training_state(self, opt, epoch):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
        """
        if epoch != -1:
            state = {
                'epoch': epoch,
                'optim': self.optim.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler else None
            }
            save_filename = f'epoch_{epoch}.state'
            save_path = os.path.join(opt['exp']['exp_root'], 'states',
                                     save_filename)
            make_dir(os.path.dirname(save_path))
            torch.save(state, save_path)

    def resume_training(self, state_path):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        resume_state = torch.load(state_path, map_location='cpu')
        resume_optimizer = resume_state['optim']
        resume_scheduler = resume_state['scheduler']
        self.optim.load_state_dict(resume_optimizer)
        if self.scheduler:
            self.scheduler.load_state_dict(resume_scheduler)
        return resume_state['epoch']

    def reduce_dict(self, inp_dict, reduction='mean'):
        """reduce  inp_dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
            inp_dict (OrderedDict): dict to be reduce in all ranks.
            reduction: return mean or sum between or gpus
        """
        with torch.no_grad():
            keys = []
            values = []
            for name, value in inp_dict.items():
                keys.append(name)
                values.append(value)
            values = torch.stack(values, 0)
            torch.distributed.reduce(values, dst=0)
            if reduction == 'mean' and self.rank == 0:
                values /= self.total_rank
            reduce_dict = {key: value.item() for key, value in zip(keys, values)}
        # log_dict = OrderedDict()
        # for name, value in loss_dict.items():
            # log_dict[name] = value.item()
        return reduce_dict

    def dict_wrapper(self, inp_dict):
        out_dict = {}
        for k,v in inp_dict.items():
            out_dict[k] = v.item()
        return out_dict
    
    def init_weight(module_list, init_type='kaiming', bia_fill=0):
        pass