import os.path as osp
import time
import cPickle as pickle
import numpy as np
from utils.cython_bbox import bbox_overlaps
import matplotlib.pyplot as plt

BBOX_MIN_OVERLAP = 0.5
EPS = 1e-14


def orochi_eval(classes, boxes_by_classes, gtb_obj, output_dir):
    """
    create results obj from detections, save it into pkl file,
    show results.
    """
    results = []
    for i, cls in enumerate(classes):
        boxes_info = boxes_by_classes[i]
        gt_ids, gt_boxes = _create_gt_info_for_cls(
            cls, boxes_info, gtb_obj)
        result = _eval_cls(cls, boxes_info, gt_ids, gt_boxes)
        results.append(result)

    path = osp.join(output_dir, 'eval_result.pkl')
    with open(path, 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    _save_precision_per_recall_img(results, output_dir)
    _print_eval_results(results)


def print_eval_results(output_dir):
    """
    show results from pre-computed results file.
    """
    path = osp.join(output_dir, 'eval_result.pkl')
    if not osp.isfile(path):
        raise IOError(('{:s} not found.').format(path))

    with open(path, 'rb') as f:
        results = pickle.load(f)

    _print_eval_results(results)


def _create_gt_info_for_cls(cls, boxes_info, gtb_obj):
    img_ids, _ = boxes_info

    img_id_len = len(img_ids[0])
    gt_ids = np.zeros(0, dtype=np.str)
    gt_boxes = np.zeros((0, 4), dtype=np.float)

    uniq_img_ids = np.unique(img_ids)
    for i in xrange(uniq_img_ids.size):
        img_id = uniq_img_ids[i]
        gt_boxes_for_img = gtb_obj['boxes'][img_id]
        gt_labels_for_img = gtb_obj['labels'][img_id]
        inds = np.where(gt_labels_for_img == cls)[0]
        if inds.size == 0:
            continue

        _gt_ids = np.empty(inds.size, dtype=(np.str, img_id_len))
        _gt_ids.fill(img_id)
        gt_ids = np.hstack((gt_ids, _gt_ids))
        gt_boxes = np.vstack((gt_boxes, gt_boxes_for_img[inds]))

    return gt_ids, gt_boxes


def _eval_cls(cls, boxes_info, gt_ids, gt_boxes):
    img_ids, dets = boxes_info
    score = dets[:, -1]
    bbox = dets[:, 0:4]

    # sort detections by decreasing confidence (score)
    si = np.argsort(-score)
    img_ids = img_ids[si]
    bbox = bbox[si]

    # assign detections to ground truth objects
    nd = score.shape[0]
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    # display progress
    tm = time.time()
    progress = 0
    print '{:s}: pr: compute: {:d}/{:d}'.format(cls, progress, nd)

    uniq_img_ids = np.unique(img_ids)
    for img_id in uniq_img_ids:
        # retrieve bbox corresponding to the img_id
        inds = np.where(img_ids == img_id)[0]
        bbox_for_img = bbox[inds]
        # retrieve gt bbox corresponding to the img_id
        gt_inds = np.where(gt_ids == img_id)[0]
        if gt_inds.size == 0:
            fp[inds] = 1   # false positives
            continue
        gt_bbox_for_img = gt_boxes[gt_inds]
        overlaps = bbox_overlaps(bbox_for_img.astype(np.float),
                                 gt_bbox_for_img.astype(np.float))

        # for each detected bbox in the order of descending
        # confidence, pick the max-overlapped gt box.
        # mark used gt boxes, since we want to classify multiple
        # detection as false positive.
        used = np.zeros(overlaps.shape[1], dtype=np.int)
        for i in xrange(overlaps.shape[0]):
            # display progress
            progress += 1
            if time.time() - tm > 1:
                print '{:s}: pr: compute: {:d}/{:d}'.format(
                    cls, progress, nd)
                tm = time.time()

            ov = overlaps[i]
            ind = inds[i]

            ovmax_ind = ov.argmax()
            ovmax = ov.max()
            if ovmax > BBOX_MIN_OVERLAP - EPS:
                if not used[ovmax_ind]:
                    tp[ind] = 1   # true positive
                    used[ovmax_ind] = 1
                else:
                    fp[ind] = 1  # false positive (multiple detection)
            else:
                fp[ind] = 1   # false positive

    # compute precision and recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / gt_ids.size
    prec = tp / (fp + tp)

    # compute average precision
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

    return {'cls': cls, 'rec': rec, 'prec': prec, 'ap': ap}


def _save_precision_per_recall_img(results, output_dir):
    for result in results:
        fig = plt.figure()
        cls = result['cls'].decode('utf-8')
        fig.suptitle(u'class: {:s}, subset: test, AP = {:.3f}'.format(
            cls, result['ap']))
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.plot(result['rec'], result['prec'], '-')
        plt.draw()
        filename = osp.join(output_dir, u'{:s}_pr.jpeg'.format(cls))
        plt.savefig(filename)
        plt.close(fig)


def _print_eval_results(results):
    print '~~~~~~~~~~~~~~~~~~~~'
    print 'Results:'
    ap_sum = 0.0
    for result in results:
        ap_sum += result['ap']
        print '{:s}: {:.1f}'.format(
            result['cls'], result['ap'] * 100)
    print ''
    print 'mAP: {:.1f}'.format((ap_sum / len(results)) * 100)
    print '~~~~~~~~~~~~~~~~~~~~'
