import os
from datetime import datetime
from basicseg.utils.dist_util import master_only
import time

@master_only
def make_dir(root):
    if not os.path.exists(root):
        os.makedirs(root)

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

def make_exp_root(root):
    exp_root = os.path.join(root, get_time_str())
    make_dir(exp_root)
    return exp_root


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = os.path.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

def scandir_SIDD(dir_path, keywords=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str): Path of the directory.
        keywords (str | tuple(str), optional): File keywords that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (keywords is not None) and not isinstance(keywords, (str, tuple)):
        raise TypeError('"keywords" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, keywords, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = os.path.relpath(entry.path, root)

                if keywords is None:
                    yield return_path
                elif return_path.find(keywords) > 0:
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, keywords=keywords, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, keywords=keywords, recursive=recursive)

if __name__ == '__main__':
    time_ = get_time_str()
    print(time_)