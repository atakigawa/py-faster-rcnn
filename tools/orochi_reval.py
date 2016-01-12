#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Reval = re-eval. Re-evaluate saved detections."""

import _init_paths
from fast_rcnn.test import apply_nms
from fast_rcnn.config import cfg, cfg_from_file
from datasets.orochi_factory import get_imdb
import cPickle as pickle
import os
import sys
import argparse
import pprint
import numpy as np


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Re-evaluate results')
    parser.add_argument('output_dir', nargs=1, help='results directory',
                        type=str)
    parser.add_argument('--rerun', dest='rerun',
                        help=('re-run evaluation code '
                              '(otherwise: results are loaded from file)'),
                        action='store_true')
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to re-evaluate',
                        required=True, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def from_mats(imdb_name, output_dir):
    # TODO
    # import scipy.io as sio

    # imdb = get_imdb(imdb_name)

    # aps = []
    # for i, cls in enumerate(imdb.classes[1:]):
    #     mat = sio.loadmat(os.path.join(output_dir, cls + '_pr.mat'))
    #     ap = mat['ap'][0, 0] * 100
    #     apAuC = mat['ap_auc'][0, 0] * 100
    #     print '!!! {} : {:.1f} {:.1f}'.format(cls, ap, apAuC)
    #     aps.append(ap)

    print '~~~~~~~~~~~~~~~~~~~'
    print 'Results (from mat files):'
    for ap in aps:
        print '{:.1f}'.format(ap)
    print '{:.1f}'.format(np.array(aps).mean())
    print '~~~~~~~~~~~~~~~~~~~'


def from_dets(imdb_name, output_dir):
    imdb = get_imdb(imdb_name)
    with open(os.path.join(output_dir, 'detections.pkl'), 'rb') as f:
        dets = pickle.load(f)

    print 'Applying NMS to all detections'
    nms_dets = apply_nms(dets, cfg.TEST.NMS)

    print 'Evaluating detections'
    imdb.evaluate_detections(nms_dets, output_dir)


if __name__ == '__main__':
    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    output_dir = os.path.abspath(args.output_dir[0])

    imdb_name = args.imdb_name
    # should be called on test dataset.
    if imdb_name[-4:] != 'test':
        imdb_name += '_test'

    if args.rerun:
        from_dets(imdb_name, output_dir)
    else:
        from_mats(imdb_name, output_dir)
