import yaml
from collections import OrderedDict
import argparse

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required = True, help='Path to option YAML file.')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--device', default='0')
    parser.add_argument(
        '--force_yml', nargs='+', default=None, help='Force to update yml files. Examples: train:ema_decay=0.999')
    args = parser.parse_args()

    # parse yml to dict
    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    # opt['rank'], opt['world_size'] = get_dist_info()


    # force to update yml options
    # if args.force_yml is not None:
    #     for entry in args.force_yml:
    #         # now do not support creating new keys
    #         keys, value = entry.split('=')
    #         keys, value = keys.strip(), value.strip()
    #         value = _postprocess_yml_value(value)
    #         eval_str = 'opt'
    #         for key in keys.split(':'):
    #             eval_str += f'["{key}"]'
    #         eval_str += '=value'
    #         # using exec function
    #         exec(eval_str)

    # datasets
    # for phase, dataset in opt['datasets'].items():
    #     # for multiple datasets, e.g., val_1, val_2; test_1, test_2
    #     phase = phase.split('_')[0]
    #     dataset['phase'] = phase
    #     if 'scale' in opt:
    #         dataset['scale'] = opt['scale']
    #     if dataset.get('dataroot_gt') is not None:
    #         dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
    #     if dataset.get('dataroot_lq') is not None:
    #         dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])

    # paths
    # for key, val in opt['path'].items():
        # if (val is not None) and ('resume_state' in key or 'pretrain_network' in key):
    #         opt['path'][key] = osp.expanduser(val)

    # if is_train:
    #     experiments_root = osp.join(root_path, 'experiments', opt['name'])
    #     opt['path']['experiments_root'] = experiments_root
    #     opt['path']['models'] = osp.join(experiments_root, 'models')
    #     opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
    #     opt['path']['log'] = experiments_root
    #     opt['path']['visualization'] = osp.join(experiments_root, 'visualization')

    return opt, args