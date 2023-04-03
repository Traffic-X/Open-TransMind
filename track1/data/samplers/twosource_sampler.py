# !/usr/bin/env python3
# encoding: utf-8

import copy
import itertools
from typing import Optional
import logging

import numpy as np
from paddle.io import Sampler

from utils import comm
logger = logging.getLogger(__name__)

class TrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.
    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(self, dataset, batchsize, shuffle=True, seed=None, dp_group=None, moe_group=None):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = len(dataset)
        assert self._size > 0
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        # self._rank = comm.get_rank()
        # self._world_size = comm.get_world_size()
        if dp_group is None: 
            self._rank = comm.get_rank()
            self._world_size = comm.get_world_size()
        else:
            self._rank = comm.get_rank() // moe_group.nranks
            self._world_size = dp_group.nranks
        logger.info("dataset {}: rank {} is mapped to _rank {} under the real local world size {}".format(dataset.dataset_name, comm.get_rank(), self._rank, self._world_size))
        self.batchsize = batchsize

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)
    
    def __len__(self,):
        return 0 #len(self.local_indices)

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:

            if self._shuffle:
                yield from np.random.permutation(self._size)
            else:
                yield from np.arange(self._size)

    def _load_batch(self):
            batch = []
            if self.two_source:
                random_categories = list(np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 9, 12], self._batch_size, \
                    replace=self.is_sample_replace, p=[0.0875, 0.15, 0.15, 0.0875, 0.0875, 0.0875, 0.0875, 0.0875, 0.0875, 0.0875])) 
                    #replace=self.is_sample_replace, p=[0.0825, 0.1450, 0.1450, 0.0825, 0.0825, 0.0925, 0.0925, 0.0925, 0.0925, 0.0925]))
                for idx, cls in enumerate(random_categories):
                    #if idx % 2 == 0:
                    if idx % 5 <= 1:
                        if cls not in self.category_imgids:
                            cls = 1
                        cur_id = np.random.choice(self.category_imgids[cls], self.num_img_per_cls, replace=False)[0]
                        sample = copy.deepcopy(self._roidbs[cur_id])
                    else:
                        if cls not in self.category_imgids_2:
                            cls = 1
                        if cls == 9 or cls == 12:
                            cls = list(np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], 1, replace=False))[0]
                        cur_id = np.random.choice(self.category_imgids_2[cls], self.num_img_per_cls, replace=False)[0]
                        sample = copy.deepcopy(self._roidbs_2[cur_id])

                    if self._drop_empty and self._fields and 'gt_mask' in self._fields:
                        if self._has_empty(self._segm(sample)):
                            continue
                    if self._drop_empty and self._fields and 'gt_bbox' in self._fields:
                        while self._has_empty(sample['gt_bbox']):
                            if idx % 2 == 0:
                                cur_id = np.random.choice(self.category_imgids[cls], 1, replace=False)[0]
                                sample = copy.deepcopy(self._roidbs[cur_id])
                            else:
                                cur_id = np.random.choice(self.category_imgids_2[cls], 1, replace=False)[0]
                                sample = copy.deepcopy(self._roidbs_2[cur_id])

                    if self._load_img:
                        sample['image'] = self._load_image(sample['im_file']) 
                    batch.append(sample)