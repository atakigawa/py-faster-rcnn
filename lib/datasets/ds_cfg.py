import yaml
import os.path as osp
import datasets
from easydict import EasyDict as edict

__cfg = {}

def get_cfg(filename=None):
    global __cfg
    if __cfg:
        return __cfg

    if filename is None:
        filename = osp.join(
            datasets.ROOT_DIR, 'experiments', 'cfgs',
            'orochi_frcnn_dataset.yml')

    with open(filename, 'r') as f:
        __cfg = edict(yaml.load(f))

    return __cfg
