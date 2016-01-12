"""
python port of voc_opts.
"""

import os.path as osp
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict


def get_voc_opts(year):
    avail_years = ['2007', '2010', '2012']
    assert (year in avail_years), \
        'year should be one of: {}'.format(avail_years)

    if year == '2007':
        return _voc_opts_2007()
    if year == '2010':
        return _voc_opts_2010()
    if year == '2012':
        return _voc_opts_2012()
    else:
        return None


def _voc_opts_2007():
    VOCopts = edict()

    VOCopts.dataset = 'VOC2007'

    VOCopts.testset = 'val'  # use validation data for development test set
    # VOCopts.testset='test'   # use test set for final challenge

    devkitroot = osp.join(osp.dirname(__file__), '..', '..', '..',
            'data', 'VOCdevkit2007')
    VOCopts.resdir = osp.join(devkitroot, 'results', VOCopts.dataset)
    VOCopts.annopath = osp.join(devkitroot, VOCopts.dataset,
            'Annotations', '{}.xml')
    VOCopts.imgsetpath = osp.join(devkitroot, VOCopts.dataset,
            'ImageSets', 'Main', '{}.txt')
    VOCopts.detrespath = (lambda VOCopts=VOCopts:
        osp.join(VOCopts.resdir, 'Main',
            '{}_det_' + VOCopts.testset + '_{}.txt'))

    VOCopts.classes = [
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor'
    ]
    VOCopts.nclasses = len(VOCopts.classes)

    VOCopts.poses = [
        'Unspecified',
        'SideFaceLeft',
        'SideFaceRight',
        'Frontal',
        'Rear'
    ]
    VOCopts.nposes = len(VOCopts.poses)

    VOCopts.parts = [
        'head',
        'hand',
        'foot'
    ]
    VOCopts.maxparts = [1, 2, 2]   # max of each of above parts
    VOCopts.nparts = len(VOCopts.parts)

    VOCopts.minoverlap = 0.5

    return VOCopts


def _voc_opts_2010():
    VOCopts = edict()

    VOCopts.dataset = 'VOC2010'

    VOCopts.trainset = 'train'  # use train for development
    # VOCopts.trainset = 'trainval'  # use train+val for final challenge

    VOCopts.testset = 'val'   # use validation data for development test set
    # VOCopts.testset = 'test'  # use test set for final challenge

    devkitroot = osp.join(osp.dirname(__file__), '..', '..', '..',
            'data', 'VOCdevkit2010')
    VOCopts.resdir = osp.join(devkitroot, 'results', VOCopts.dataset)
    VOCopts.annopath = osp.join(devkitroot, VOCopts.dataset,
            'Annotations', '{}.xml')
    VOCopts.imgsetpath = osp.join(devkitroot, VOCopts.dataset,
            'ImageSets', 'Main', '{}.txt')
    VOCopts.detrespath = (lambda VOCopts=VOCopts:
        osp.join(VOCopts.resdir, 'Main',
            '{}_det_' + VOCopts.testset + '_{}.txt'))

    VOCopts.classes = [
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor'
    ]
    VOCopts.nclasses = len(VOCopts.classes)

    VOCopts.poses = [
        'Unspecified',
        'Left',
        'Right',
        'Frontal',
        'Rear'
    ]
    VOCopts.nposes = len(VOCopts.poses)

    VOCopts.parts = [
        'head',
        'hand',
        'foot'
    ]
    VOCopts.nparts = len(VOCopts.parts)
    VOCopts.maxparts = [1, 2, 2]   # max of each of above parts

    VOCopts.actions = [
        'phoning',
        'playinginstrument',
        'reading',
        'ridingbike',
        'ridinghorse',
        'running',
        'takingphoto',
        'usingcomputer',
        'walking'
    ]
    VOCopts.nactions = len(VOCopts.actions)

    VOCopts.minoverlap = 0.5


def _voc_opts_2012():
    VOCopts = edict()

    VOCopts.dataset = 'VOC2012'

    VOCopts.trainset = 'train'   # use train for development
    # VOCopts.trainset = 'trainval'  # use train+val for final challenge

    VOCopts.testset = 'val'  # use validation data for development test set
    # VOCopts.testset = 'test'  # use test set for final challenge

    devkitroot = osp.join(osp.dirname(__file__), '..', '..', '..',
            'data', 'VOCdevkit2012')
    VOCopts.resdir = osp.join(devkitroot, 'results', VOCopts.dataset)
    VOCopts.annopath = osp.join(devkitroot, VOCopts.dataset,
            'Annotations', '{}.xml')
    VOCopts.imgsetpath = osp.join(devkitroot, VOCopts.dataset,
            'ImageSets', 'Main', '{}.txt')
    VOCopts.detrespath = (lambda VOCopts=VOCopts:
        osp.join(VOCopts.resdir, 'Main',
            '{}_det_' + VOCopts.testset + '_{}.txt'))

    VOCopts.classes = [
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor'
    ]
    VOCopts.nclasses = len(VOCopts.classes)

    VOCopts.poses = [
        'Unspecified',
        'Left',
        'Right',
        'Frontal',
        'Rear'
    ]
    VOCopts.nposes = len(VOCopts.poses)

    VOCopts.parts = [
        'head',
        'hand',
        'foot'
    ]
    VOCopts.nparts = len(VOCopts.parts)
    VOCopts.maxparts = [1, 2, 2]   # max of each of above parts

    VOCopts.actions = [
        'other',             # skip this when training classifiers
        'jumping',
        'phoning',
        'playinginstrument',
        'reading',
        'ridingbike',
        'ridinghorse',
        'running',
        'takingphoto',
        'usingcomputer',
        'walking'
    ]
    VOCopts.nactions = len(VOCopts.actions)

    VOCopts.minoverlap = 0.5
