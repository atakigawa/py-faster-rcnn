"""
python port of xVOCap.
"""

import numpy as np


def x_voc_ap(rec, prec):
    # From the PASCAL VOC 2011 devkit

    # mrec=[0 ; rec ; 1];
    # mpre=[0 ; prec ; 0];
    mrec = np.concatenate(([0], rec, [1]))
    mpre = np.concatenate(([0], prec, [0]))

    # for i=numel(mpre)-1:-1:1
    #     mpre(i)=max(mpre(i),mpre(i+1));
    # end
    mpre[:-1] = np.maximum(mpre[:-1], mpre[1:])

    # i=find(mrec(2:end)~=mrec(1:end-1))+1;
    i = np.where(mrec[1:] != mrec[:-1])[0] + 1

    # ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i])

    return ap
