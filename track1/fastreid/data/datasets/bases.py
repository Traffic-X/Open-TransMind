# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import copy
import logging
import os
import paddle
from typing import Sequence

from tabulate import tabulate
from termcolor import colored
import numpy as np
import random

logger = logging.getLogger(__name__)


class Dataset(object):
    """An abstract class representing a Dataset.
    This is the base class for ``ImageDataset`` and ``VideoDataset``.

    Args:
        train (list or Callable): contains tuples of (img_path(s), pid, camid).
        query (list or Callable): contains tuples of (img_path(s), pid, camid).
        gallery (list or Callable): contains tuples of (img_path(s), pid, camid).
        transform: transform function.
        mode (str): 'train', 'query' or 'gallery'.
        combineall (bool): combines train, query and gallery in a
            dataset for training.
        verbose (bool): show information.
    """
    _junk_pids = []  # contains useless person IDs, e.g. background, false detections

    def __init__(self, train, query, gallery, transform=None, mode='train',
                 combineall=False, verbose=True, **kwargs):
        self._train = train
        self._query = query
        self._gallery = gallery
        self.transform = transform
        self.mode = mode
        self.combineall = combineall
        self.verbose = verbose

        if self.combineall:
            self.combine_all()

        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'query':
            self.data = self.query
        elif self.mode == 'gallery':
            self.data = self.gallery
        else:
            raise ValueError('Invalid mode. Got {}, but expected to be '
                             'one of [train | query | gallery]'.format(self.mode))

    @property
    def train(self):
        """train
        """
        if callable(self._train):
            self._train = self._train()
        return self._train

    @property
    def query(self):
        """query
        """
        if callable(self._query):
            self._query = self._query()
        return self._query

    @property
    def gallery(self):
        """gallery
        """
        if callable(self._gallery):
            self._gallery = self._gallery()
        return self._gallery

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __radd__(self, other):
        """Supports sum([dataset1, dataset2, dataset3])."""
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def parse_data(self, data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.
        Args:
            data (list): contains tuples of (img_path(s), pid, camid)
        """
        pids = set()
        cams = set()
        for info in data:
            pids.add(info[1])
            cams.add(info[2])
        return len(pids), len(cams)

    def get_num_pids(self, data):
        """Returns the number of training person identities."""
        return self.parse_data(data)[0]

    def get_num_cams(self, data):
        """Returns the number of training cameras."""
        return self.parse_data(data)[1]

    def show_summary(self):
        """Shows dataset statistics."""
        pass

    def combine_all(self):
        """Combines train, query and gallery in a dataset for training."""
        combined = copy.deepcopy(self.train)

        def _combine_data(data):
            for img_path, pid, camid in data:
                if pid in self._junk_pids:
                    continue
                pid = getattr(self, "dataset_name", "Unknown") + "_test_" + str(pid)
                camid = getattr(self, "dataset_name", "Unknown") + "_test_" + str(camid)
                combined.append((img_path, pid, camid))

        _combine_data(self.query)
        _combine_data(self.gallery)

        self._train = combined

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.
        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not os.path.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))


class ImageDataset(Dataset):
    """A base class representing ImageDataset.
    All other image datasets should subclass it.
    ``__getitem__`` returns an image given index.
    It will return ``img``, ``pid``, ``camid`` and ``img_path``
    where ``img`` has shape (channel, height, width). As a result,
    data in each batch has shape (batch_size, channel, height, width).
    """

    def show_train(self):
        """show_train
        """
        num_train_pids, num_train_cams = self.parse_data(self.train)

        headers = ['subset', '# ids', '# images', '# cameras']
        csv_results = [['train', num_train_pids, len(self.train), num_train_cams]]

        # tabulate it
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )
        logger.info("=> Loaded {} in csv format: \n".format(self.__class__.__name__) + colored(table, "cyan"))

    def show_test(self):
        """show_test
        """
        num_query_pids, num_query_cams = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)

        headers = ['subset', '# ids', '# images', '# cameras']
        csv_results = [
            ['query', num_query_pids, len(self.query), num_query_cams],
            ['gallery', num_gallery_pids, len(self.gallery), num_gallery_cams],
        ]

        # tabulate it
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )
        logger.info("=> Loaded {} in csv format: \n".format(self.__class__.__name__) + colored(table, "cyan"))


class DetDataset(paddle.io.Dataset):
    """
    Load detection dataset.

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): annotation file path.
        data_fields (list): key name of data dictionary, at least have 'image'.
        sample_num (int): number of samples to load, -1 means all.
        use_default_label (bool): whether to load default label list.
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1,
                 use_default_label=None,
                 **kwargs):
        super(DetDataset, self).__init__()
        self.dataset_dir = dataset_dir if dataset_dir is not None else ''
        self.anno_path = anno_path
        self.image_dir = image_dir if image_dir is not None else ''
        self.data_fields = data_fields
        self.sample_num = sample_num
        self.use_default_label = use_default_label
        self._epoch = 0
        self._curr_iter = 0

    def __len__(self, ):
        return len(self.roidbs)

    def __call__(self, *args, **kwargs):
        return self

    def __getitem__(self, idx):
        # data batch
        roidb = copy.deepcopy(self.roidbs[idx])
        if self.mixup_epoch == 0 or self._epoch < self.mixup_epoch:
            n = len(self.roidbs)
            idx = np.random.randint(n)
            roidb = [roidb, copy.deepcopy(self.roidbs[idx])]
        elif self.cutmix_epoch == 0 or self._epoch < self.cutmix_epoch:
            n = len(self.roidbs)
            idx = np.random.randint(n)
            roidb = [roidb, copy.deepcopy(self.roidbs[idx])]
        elif self.mosaic_epoch == 0 or self._epoch < self.mosaic_epoch:
            n = len(self.roidbs)
            roidb = [roidb, ] + [
                copy.deepcopy(self.roidbs[np.random.randint(n)])
                for _ in range(4)
            ]
        if isinstance(roidb, Sequence):
            for r in roidb:
                r['curr_iter'] = self._curr_iter
        else:
            roidb['curr_iter'] = self._curr_iter
        self._curr_iter += 1

        return self.transform(roidb)

    def set_kwargs(self, **kwargs):
        self.mixup_epoch = kwargs.get('mixup_epoch', -1)
        self.cutmix_epoch = kwargs.get('cutmix_epoch', -1)
        self.mosaic_epoch = kwargs.get('mosaic_epoch', -1)

    def set_transform(self, transform):
        self.transform = transform

    def set_epoch(self, epoch_id):
        self._epoch = epoch_id

    def parse_dataset(self, ):
        raise NotImplementedError(
            "Need to implement parse_dataset method of Dataset")

    def get_anno(self):
        if self.anno_path is None:
            return
        return os.path.join(self.dataset_dir, self.anno_path)
