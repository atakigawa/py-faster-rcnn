"""
python port of
VOCdevkit2007/VOCcode/VOCevaldet.m
"""

import numpy as np
import time
from utils.cython_bbox import bbox_overlaps
import matplotlib.pyplot as plt
import VOCreadrecxml


def voc_eval_det(VOCopts, comp_id, cls, draw):
    # load test set
    # [gtids,t]=textread(sprintf(VOCopts.imgsetpath,VOCopts.testset),'%s %d');
    with open(VOCopts.imgsetpath.format(VOCopts.testset)) as f:
        gtids = np.loadtxt(f, dtype=np.str)

    # load ground truth objects
    tm = time.time()
    npos = 0
    # gt(length(gtids))=struct('BB',[],'diff',[],'det',[]);
    len_gtids = gtids.size
    gt = np.array([{'BB':[], 'diff':[], 'det':[]} for _ in xrange(len_gtids)])
    for i in xrange(len_gtids):
        # display progress
        if time.time() - tm > 1:
            print '{:s}: pr: load: {:d}/{:d}'.format(cls, i + 1, len_gtids)
            tm = time.time()

        # read annotation
        rec = VOCreadrecxml.voc_read_rec_xml(VOCopts.annopath.format(gtids[i]))

        # extract objects of class
        # clsinds=strmatch(cls,{rec.objects(:).class},'exact');
        # gt(i).BB=cat(1,rec.objects(clsinds).bbox)';
        # gt(i).diff=[rec.objects(clsinds).difficult];
        # gt(i).det=false(length(clsinds),1);
        # npos=npos+sum(~gt(i).diff);
        objects = rec.findall('objects')
        bb = []
        diff = []
        det = []
        for obj in objects:
            if obj.find('class').text != cls:
                continue
            bb.append(obj.find('bbox').text)
            diff.append(obj.find('difficult').text)
            det.append(False)

        gt[i]['BB'] = np.array(bb)
        gt[i]['diff'] = np.array(diff)
        gt[i]['det'] = np.array(det)
        # add number of non-difficult boxes
        npos += np.sum(np.logical_not(gt[i]['diff']))

    # load results
    # [ids,confidence,b1,b2,b3,b4]=
    #   textread(sprintf(VOCopts.detrespath,id,cls),'%s %f %f %f %f %f');
    with open(VOCopts.detrespath().format(comp_id, cls)) as f:
        dets = np.loadtxt(f, dtype=[('id', np.str, 6),
                                    ('confidence', np.float),
                                    ('xmin', np.float),
                                    ('ymin', np.float),
                                    ('xmax', np.float),
                                    ('ymax', np.float)])
    ids = dets['id']
    confidence = dets['confidence']
    _BB = dets[['xmin', 'ymin', 'xmax', 'ymax']]
    BB = _BB.view(dtype=(np.float, 4))

    # sort detections by decreasing confidence
    si = np.argsort(-confidence)
    ids = ids[si]
    BB = BB[si]

    # assign detections to ground truth objects
    nd = len(confidence)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    tm = time.time()
    for d in xrange(nd):
        # display progress
        if time.time() - tm > 1:
            print '{:s}: pr: compute: {:d}/{:d}'.format(cls, d + 1, nd)
            tm = time.time()

        # find ground truth image
        inds = np.where(gtids == ids[d])[0]
        assert inds.size > 0, 'unrecognized image "%s"'.format(ids[d])
        assert inds.size < 2, 'multiple image "%s"'.format(ids[d])
        target_gt = gt[inds[0]]

        # assign detection to ground truth object if any
        # ovmax=-inf;
        # for j=1:size(gt(i).BB,2)
        #     bbgt=gt(i).BB(:,j);
        #     bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ;
        #         min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
        #     iw=bi(3)-bi(1)+1;
        #     ih=bi(4)-bi(2)+1;
        #     if iw>0 & ih>0
        #         % compute overlap as area of intersection / area of union
        #         ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
        #            (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
        #            iw*ih;
        #         ov=iw*ih/ua;
        #         if ov>ovmax
        #             ovmax=ov;
        #             jmax=j;
        #         end
        #     end
        # end
        #
        # assign detection as true positive/don't care/false positive
        # if ovmax>=VOCopts.minoverlap
        #     if ~gt(i).diff(jmax)
        #         if ~gt(i).det(jmax)
        #             tp(d)=1;            % true positive
        #             gt(i).det(jmax)=true;
        #         else
        #             fp(d)=1;            % false positive (multiple detection)
        #         end
        #     end
        # else
        #     fp(d)=1;                    % false positive
        # end

        if target_gt['BB'].size == 0:
            fp[d] = 1   # false positive
            continue

        bb = BB[d]
        overlaps = bbox_overlaps(bb[np.newaxis, :], target_gt['BB'])
        ov = overlaps[0]
        ovmax_ind = np.argmax(ov)
        ovmax = np.max(ov)

        if ovmax >= VOCopts.minoverlap:
            if not target_gt['diff'][ovmax_ind]:
                if not target_gt['det'][ovmax_ind]:
                    tp[d] = 1  # true positive
                    target_gt['det'][ovmax_ind] = True
                else:
                    fp[d] = 1  # false positive (multiple detection)
        else:
            fp[d] = 1   # false positive

    # compute precision/recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / npos
    prec = tp / (fp + tp)

    # compute average precision
    # ap=0;
    # for t=0:0.1:1
    #     p=max(prec(rec>=t));
    #     if isempty(p)
    #         p=0;
    #     end
    #     ap=ap+p/11;
    # end
    ap = 0
    for t in xrange(11):
        t /= 10.0
        ind = np.where(rec > t - 1e-5)[0]
        ps = prec[ind]
        if ps.size == 0:
            p = 0
        else:
            p = np.max(prec[ind])
        ap = ap + (p / 11)

    if draw:
        # plot precision/recall
        # plot(rec,prec,'-');
        # grid;
        # xlabel 'recall'
        # ylabel 'precision'
        # title(sprintf(
        #   'class: %s, subset: %s, AP = %.3f',cls,VOCopts.testset,ap));
        fig = plt.figure()
        fig.suptitle('class: {}, subset: {}, AP = {:.3f}'.format(
            cls, VOCopts.testset, ap))
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.plot(rec, prec, '-')
        plt.draw()

    return rec, prec, ap
