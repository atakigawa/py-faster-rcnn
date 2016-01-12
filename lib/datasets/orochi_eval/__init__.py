# import os
# import os.path as osp
# import sys
# import numpy as np
# import matplotlib.pyplot as plt


def orochi_eval(classes, boxes_by_classes, gtb_obj):
    print 'orochi_eval'
    print len(classes)
    for arr in boxes_by_classes:
        print arr[0].shape
        print arr[1].shape
    # gtb_obj['boxes'][img_idx]
    # gtb_obj['labels'][img_idx]

    # TODO
