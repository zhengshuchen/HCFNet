import importlib
from os import path as osp
from copy import deepcopy
from basicseg.utils.path_utils import scandir
from basicseg.utils.registry import LOSS_REGISTRY
__all__ = ['build_loss']

arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder)
    if v.endswith('.py')
]
# import all the arch modules
_arch_modules = [
    importlib.import_module(f'basicseg.loss.{file_name}')
    for file_name in arch_filenames
]

def build_loss(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    opt.pop('weight')
    net = LOSS_REGISTRY.get(network_type)(**opt)
    # logger = get_root_logger()
    # logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net

def main():
    opt = {'type':'Bce_loss'}
    loss = build_loss(opt)
    import torch
    pred = torch.rand(2,1,512,512)
    mask = torch.rand(2,1,512,512)
    print(loss(pred, mask))

if __name__ == '__main__':
    main()
