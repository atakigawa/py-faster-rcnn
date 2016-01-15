#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import caffe
import os.path as osp
import cv2
import argparse
from datasets.orochi_factory import get_imdb
import datasets.ds_cfg
from datetime import datetime


def viz_detections(im, cls_inds, dets, ind_to_cls, thresh):
    """Draw detected bounding boxes."""

    # BGR -> RGB
    im = im[:, :, (2, 1, 0)]

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in xrange(cls_inds.size):
        cls_ind = cls_inds[i]
        cls = ind_to_cls[cls_ind].decode('utf-8')
        bbox = dets[i, :4]
        score = dets[i, -1]

        if score < thresh:
            continue

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5))
        ax.text(bbox[0], bbox[1] - 6,
                # u'{:s} {:.2f}'.format(cls, score),
                u'{:s}'.format(cls),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=8, color='white')

    ax.set_title(('detections with (thresh >= {}:.1f)').format(
        thresh), fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(net, imdb, image_path, save_img):
    """
    Detect object classes in an image using pre-computed object proposals.
    """

    # Load the demo image
    im = cv2.imread(image_path)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections
    CONF_THRESH = 0.3
    # low NMS_THRESH rate is OK for OCR use case.
    NMS_THRESH = 0.1
    # skip background
    cls_inds = np.zeros(0, dtype=np.int)
    dets = np.zeros((0, 5), dtype=np.float32)
    for cls_ind, _ in enumerate(imdb.classes[1:], 1):
        cls_boxes = boxes[:, (4 * cls_ind):(4 * (cls_ind + 1))]
        cls_scores = scores[:, cls_ind]

        # sort in descending order of score
        si = np.argsort(-cls_scores)
        cls_boxes = cls_boxes[si]
        cls_scores = cls_scores[si]

        _dets = np.hstack((cls_boxes,
                           cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(_dets, NMS_THRESH)
        _dets = _dets[keep, :]
        _cls_inds = np.empty(_dets.shape[0], dtype=np.int)
        _cls_inds.fill(cls_ind)
        cls_inds = np.hstack((cls_inds, _cls_inds))
        dets = np.vstack((dets, _dets))

    ind_to_cls = dict([(i, cls) for i, cls in enumerate(imdb.classes)])
    viz_detections(im, cls_inds, dets, ind_to_cls, CONF_THRESH)

    if save_img:
        basename = osp.basename(image_path)
        filename, ext = osp.splitext(basename)
        ts = '{:%Y%m%d%H%M%S}'.format(datetime.now())
        output_filename = '{}_{}{}'.format(filename, ts, ext)
        plt.savefig(output_filename)
        print 'Saved result to {}'.format(output_filename)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Orochi Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--ds-name', dest='ds_name',
                        help='dataset name to test',
                        required=True, type=str)
    parser.add_argument('--prototxt-path', dest='prototxt_path',
                        help='prototxt path',
                        required=True, type=str)
    parser.add_argument('--save', dest='save_img',
                        help='save result image to local',
                        action='store_true')
    parser.add_argument('image_paths', nargs='*', help='results directory',
                        type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    dss = datasets.ds_cfg.get_cfg().AVAILABLE_DATASETS
    caffemodel = dss[args.ds_name].test_model_file_name
    prototxt = args.prototxt_path
    img_paths = args.image_paths

    if not osp.isfile(caffemodel):
        raise IOError(('{:s} not found.').format(caffemodel))
    if not osp.isfile(prototxt):
        raise IOError(('{:s} not found.').format(prototxt))
    for img_path in img_paths:
        if not osp.isfile(img_path):
            raise IOError(('{:s} not found.').format(img_path))

    imdb_name = args.ds_name + '_test'
    imdb = get_imdb(imdb_name)

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(net, im)

    for img_path in img_paths:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {}'.format(img_path)
        demo(net, imdb, img_path, args.save_img)

    plt.show()
