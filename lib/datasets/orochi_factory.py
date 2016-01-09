# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import datasets
from datasets.orochi_dataset import orochi_dataset
import datasets.ds_cfg
import numpy as np
from easydict import EasyDict as edict


def init():
    cfg = datasets.ds_cfg.get_cfg()
    for ds_name, _ in cfg.AVAILABLE_DATASETS.items():
        for image_set in ['train', 'test']:
            name = '{}_{}'.format(ds_name, image_set)
            __sets[name] = (lambda ds_name=ds_name, image_set=image_set:
                orochi_dataset(ds_name, image_set, cfg))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets:
        init()
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
