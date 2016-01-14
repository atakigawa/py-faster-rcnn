#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Faster R-CNN network."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.orochi_factory import get_imdb
import datasets.imdb
import datasets.ds_cfg
import caffe
import argparse
import pprint
import numpy as np
import sys
import os.path as osp
from datetime import datetime


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description='Train a Faster R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--prototxt-root', dest='prototxt_root',
                        help='prototxt root',
                        default=None, type=str)
    parser.add_argument('--prototxt-suffix', dest='prototxt_suffix',
                        help='optional prototxt suffix',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--ds-name', dest='ds_name',
                        help='dataset to train on',
                        required=True, type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def combined_roidb(imdb_names):
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb


def generate_prototxts(imdb, args):
    suffix = '{0:%Y-%m-%d_%H-%M-%S}'.format(datetime.now()) if \
        args.prototxt_suffix is None else args.prototxt_suffix

    dss = datasets.ds_cfg.get_cfg().AVAILABLE_DATASETS
    prototxt_file_dir = dss[args.ds_name].prototxt_file_dir_name
    solver_template_path = osp.join(
        args.prototxt_root, prototxt_file_dir, 'solver_template.prototxt')
    train_template_path = osp.join(
        args.prototxt_root, prototxt_file_dir, 'train_template.prototxt')
    solver_prototxt_path = osp.join(
        args.prototxt_root, prototxt_file_dir, 'train_gen',
        'solver_{}.prototxt'.format(suffix))
    train_prototxt_path = osp.join(
        args.prototxt_root, prototxt_file_dir, 'train_gen',
        'train_{}.prototxt'.format(suffix))

    with open(solver_template_path, 'r') as f:
        solver_txt = f.read()
    with open(train_template_path, 'r') as f:
        train_txt = f.read()

    solver_txt = solver_txt.replace(
        '{{train_prototxt_path}}', train_prototxt_path)

    train_txt = train_txt.replace(
        '{{num_classes}}', str(imdb.num_classes))
    train_txt = train_txt.replace(
        '{{num_bbox_pred_output}}', str(imdb.num_classes * 4))

    with open(solver_prototxt_path, 'w') as f:
        f.write(solver_txt)
    with open(train_prototxt_path, 'w') as f:
        f.write(train_txt)

    return solver_prototxt_path


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    imdb_name = args.ds_name + '_train'
    imdb, roidb = combined_roidb(imdb_name)
    print '{:d} roidb entries'.format(len(roidb))

    solver_prototxt = generate_prototxts(imdb, args)
    print('solver prototxt: {}'.format(solver_prototxt))

    output_dir = get_output_dir(imdb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    train_net(solver_prototxt, roidb, output_dir,
             pretrained_model=args.pretrained_model,
             max_iters=args.max_iters)
