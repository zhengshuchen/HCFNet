import importlib
from os import path as osp
from copy import deepcopy
from basicseg.utils.path_utils import scandir
from basicseg.utils.registry import NET_REGISTRY
from basicseg.networks.common import CDC_conv
import torch.nn as nn
__all__ = ['build_network']

arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder)
    if v.endswith('.py')
]
# import all the arch modules
_arch_modules = [
    importlib.import_module(f'basicseg.networks.{file_name}')
    for file_name in arch_filenames
]

def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = NET_REGISTRY.get(network_type)(**opt)
    # logger = get_root_logger()
    # logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net


def main():
    opt = {'type':'Fpn_res18'}
    net = build_network(opt)
    print(net)

# if __name__ == '__main__':
#     main()
