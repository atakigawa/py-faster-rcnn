"""
python port of voc_eval.
"""

import _init_paths
_init_paths.add_paths()
import scipy.io as sio
import os
import os.path as osp
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import voc_opts
import xVOCap
import VOCevaldet


def voc_eval(year, comp_id, test_set, output_dir, rm_res):
    VOCopts = voc_opts.get_voc_opts(year)
    VOCopts.testset = test_set

    res = [voc_eval_cls(cls, VOCopts, comp_id, output_dir, rm_res)
            for cls in VOCopts.classes]

    print '~~~~~~~~~~~~~~~~~~~~'
    print 'Results:'
    aps = [_res['ap'] for _res in res]
    for ap in aps:
        print '{:.1f}'.format(ap * 100)
    print '{:.1f}'.format(np.array(aps).mean() * 100)
    print '~~~~~~~~~~~~~~~~~~~~'


def voc_eval_cls(cls, VOCopts, comp_id, output_dir, rm_res):
    test_set = VOCopts.testset
    year = VOCopts.dataset[-4:]

    res_fn = VOCopts.detrespath().format(comp_id, cls)

    recall = []
    prec = []
    ap = 0
    ap_auc = 0

    do_eval = (int(year) <= 2007) or (test_set != 'test')
    if do_eval:
        recall, prec, ap = VOCevaldet.voc_eval_det(VOCopts, comp_id, cls, True)
        ap_auc = xVOCap.x_voc_ap(recall, prec)

        # % force plot limits
        # ylim([0 1]);
        # xlim([0 1]);
        #
        # print(gcf, '-djpeg', '-r0', ...
        #       [output_dir '/' cls '_pr.jpg']);
        # save img.
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        filename = osp.join(output_dir, '{}_pr.jpeg'.format(cls))
        plt.savefig(filename)
        plt.close('all')

    print '!!! {:s} : {:.4f} {:.4f}'.format(cls, ap, ap_auc)

    res = {}
    res['recall'] = recall
    res['prec'] = prec
    res['ap'] = ap
    res['ap_auc'] = ap_auc

    sio.savemat(osp.join(output_dir, '{}_pr.mat'.format(cls)), {
        'res': res,
        'recall': recall,
        'prec': prec,
        'ap': ap,
        'ap_auc': ap_auc
    })

    if rm_res:
        os.remove(res_fn)

    return res


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='do voc eval')
    parser.add_argument('year', help='VOC year', type=str)
    parser.add_argument('comp_id', help='comp id', type=str)
    parser.add_argument('test_set', help='test set', type=str)
    parser.add_argument('out_dir', help='out dir', type=str)
    parser.add_argument('rm_results', help='out dir', type=int)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    voc_eval(args.year, args.comp_id, args.test_set,
            args.out_dir, args.rm_results)
