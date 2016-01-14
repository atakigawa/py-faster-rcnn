# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
import datasets.imdb
import PIL
import numpy as np
import scipy.sparse
import cPickle as pickle
import datasets.orochi_eval as orochi_eval


class orochi_dataset(datasets.imdb):
    def __init__(self, ds_name, image_set, cfg):
        datasets.imdb.__init__(self, '{}_{}'.format(ds_name, image_set))
        self._ds_name = ds_name
        self._image_set = image_set
        self._data_path = os.path.join(cfg.DS_ROOT_DIR, ds_name)

        self._classes, self._class_to_ind = self._load_class_info()

        self._image_index = self._load_image_set_index()
        self._image_ext = '.png'

        self._roidb_handler = self.selective_search_roidb

        assert os.path.exists(self._data_path), \
                'data path does not exist: {}'.format(self._data_path)

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # e.g.
        # self._data_path + /train_list.txt
        image_set_file = os.path.join(
            self._data_path, self._image_set + '_list.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = np.loadtxt(f, dtype=np.str)
        return image_index

    def _image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # e.g.
        # self._data_path + /imgs/000003.png
        image_path = os.path.join(self._data_path, 'imgs',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._image_path_from_index(self._image_index[i])

    def _load_class_info(self):
        gtb_obj_path = os.path.join(self._data_path, 'gtb_obj.pkl')
        assert os.path.exists(gtb_obj_path), \
                'Path does not exist: {}'.format(gtb_obj_path)
        with open(gtb_obj_path, 'rb') as fid:
            gtb_obj = pickle.load(fid)

        # background always has index 0.
        classes = np.concatenate((['__background__'], gtb_obj['label_set']))
        class_to_ind = dict(zip(classes, xrange(len(classes))))
        return classes, class_to_ind

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up
        future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self._load_gt_roidb()

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _load_gt_roidb(self):
        gtb_obj_path = os.path.join(self._data_path, 'gtb_obj.pkl')
        assert os.path.exists(gtb_obj_path), \
                'Path does not exist: {}'.format(gtb_obj_path)
        with open(gtb_obj_path, 'rb') as fid:
            gtb_obj = pickle.load(fid)

        def create_roi_record(img_idx):
            boxes = gtb_obj['boxes'][img_idx]
            labels = gtb_obj['labels'][img_idx]

            num_objs = labels.shape[0]
            boxes = boxes.astype(np.uint16)
            gt_classes = np.zeros(labels.shape, dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

            for i in xrange(num_objs):
                cls_ind = self._class_to_ind[labels[i]]
                gt_classes[i] = cls_ind
                overlaps[i, cls_ind] = 1.0

            # will be inflated in lib/roi_data_layer/roidb.py
            overlaps = scipy.sparse.csr_matrix(overlaps)

            return {'boxes': boxes,
                    'gt_classes': gt_classes,
                    'gt_overlaps': overlaps,
                    'flipped': False}

        return [create_roi_record(index) for index in self.image_index]

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up
        future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        raise NotImplementedError
        # filename = os.path.abspath(os.path.join(self.cache_path, '..',
        #                                         'selective_search_data',
        #                                         self.name + '.mat'))
        # assert os.path.exists(filename), \
        #        'Selective search data not found at: {}'.format(filename)
        # raw_data = sio.loadmat(filename)['boxes'].ravel()

        # box_list = []
        # for i in xrange(raw_data.shape[0]):
        #     box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        # return self.create_roidb_from_box_list(box_list, gt_roidb)

    def rpn_roidb(self):
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _conv_all_boxes_for_eval(self, all_boxes):
        clss = self.classes[1:]  # remove __background__
        img_id_len = len(self.image_index[0])
        ret = [() for _ in clss]
        for i, cls in enumerate(clss):
            cls_ind = self._class_to_ind[cls]
            img_ids = np.zeros(0, dtype=(np.str, img_id_len))
            dets = np.zeros((0, 5), dtype=np.float)
            for im_ind, im_id in enumerate(self.image_index):
                dets_in_img = all_boxes[cls_ind][im_ind]
                if dets_in_img == []:
                    continue

                img_id_dup = np.empty(dets_in_img.shape[0],
                                      dtype=(np.str, img_id_len))
                img_id_dup.fill(im_id)
                img_ids = np.hstack((img_ids, img_id_dup))
                dets = np.vstack((dets, dets_in_img))

            ret[i] = (img_ids, dets)
        return clss, ret

    def evaluate_detections(self, all_boxes, output_dir):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """

        gtb_obj_path = os.path.join(self._data_path, 'gtb_obj.pkl')
        assert os.path.exists(gtb_obj_path), \
                'Path does not exist: {}'.format(gtb_obj_path)
        with open(gtb_obj_path, 'rb') as fid:
            gtb_obj = pickle.load(fid)

        clss, boxes = self._conv_all_boxes_for_eval(all_boxes)
        orochi_eval.orochi_eval(clss, boxes, gtb_obj, output_dir)

    def append_flipped_images(self):
        num_images = self.num_images
        widths = [PIL.Image.open(self.image_path_at(i)).size[0]
                  for i in xrange(num_images)]
        for i in xrange(num_images):
            width = widths[i]
            adjust = width % 2   # adjust when width is odd
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = np.maximum(0, width - oldx2 - adjust)
            boxes[:, 2] = np.maximum(0, width - oldx1 - adjust)
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            entry = {'boxes': boxes,
                     'gt_overlaps': self.roidb[i]['gt_overlaps'],
                     'gt_classes': self.roidb[i]['gt_classes'],
                     'flipped': True}
            self.roidb.append(entry)
        self._image_index = self._image_index * 2
