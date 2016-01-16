#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Faster R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.orochi_factory import get_imdb
import datasets.ds_cfg
import caffe
import argparse
import pprint
import sys
import os
import os.path as osp
from datetime import datetime
import jinja2


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Faster R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--prototxt-root', dest='prototxt_root',
                        help='prototxt root',
                        default=None, type=str)
    parser.add_argument('--prototxt-suffix', dest='prototxt_suffix',
                        help='optional prototxt suffix',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--ds-name', dest='ds_name',
                        help='dataset to test',
                        required=True, type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def generate_prototxts(imdb, args):
    suffix = '{0:%Y-%m-%d_%H-%M-%S}'.format(datetime.now()) if \
        args.prototxt_suffix is None else args.prototxt_suffix

    dss = datasets.ds_cfg.get_cfg().AVAILABLE_DATASETS
    prototxt_file_dir = dss[args.ds_name].prototxt_file_dir_name
    test_template_path = osp.join(
        args.prototxt_root, prototxt_file_dir, 'test_template.prototxt')
    test_prototxt_path = osp.join(
        args.prototxt_root, prototxt_file_dir, 'test_gen',
        'test_{}.prototxt'.format(suffix))

    with open(test_template_path, 'r') as f:
        test_txt = f.read()

    test_txt = jinja2.Template(test_txt).render(
        num_classes=imdb.num_classes,
        num_bbox_pred_output=(imdb.num_classes * 4)
    )

    with open(test_prototxt_path, 'w') as f:
        f.write(test_txt)

    return test_prototxt_path


def check_display():
    assert (os.environ.get('DISPLAY') is not None), \
        'DISPLAY env variable has to be exported.'


if __name__ == '__main__':
    check_display()
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

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    imdb_name = args.ds_name + '_test'
    imdb = get_imdb(imdb_name)
    imdb.competition_mode(args.comp_mode)
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    test_prototxt = generate_prototxts(imdb, args)
    print('test prototxt: {}'.format(test_prototxt))

    net = caffe.Net(test_prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    test_net(net, imdb)
