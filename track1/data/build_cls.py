import os
import random
import logging
from collections.abc import Mapping

import numpy as np
import paddle

from utils import comm
from fastreid.data import samplers
from fastreid.data.data_utils import read_image
from fastreid.data.datasets import DATASET_REGISTRY
from tools import moe_group_utils
from paddle.io import Dataset
from data.build import fast_batch_collator

from data.samplers.clsaware_reader import VehicleMultiTaskClassAwareSampler
from .datasets.fgvc_dataset import *

_root = os.getenv("FASTREID_DATASETS", "datasets")


class HierachicalCommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=False, dataset_name=None, num_classes=1000, is_train=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel
        self.dataset_name = dataset_name
        self._num_classes = num_classes
        self.labels = []
        self.id2imgname = self.id2name(img_items)
        self.is_train = is_train

        cam_set = set()
        self.cams = sorted(list(cam_set))
        if relabel:
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def id2name(self, img_items):
        id2name = {}
        for i, item in enumerate(img_items):
             img_path = item[0]
             id2name[i] = img_path
        return id2name

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        n_retry = 10
        for _ in range(n_retry):
            try:
                img_item = self.img_items[index]
                img_path = img_item[0]
                pid = img_item[1]
                camid = img_item[2]
                im_id = img_item[3]
                img = read_image(img_path)
                ori_h, ori_w, _ = np.array(img).shape
                if self.transform is not None: img = self.transform(img)
                _, h, w = img.shape
                im_shape = np.array((h, w), 'float32')
                scale_factor = np.array((h / ori_h, w / ori_w), 'float32')
                break
            except Exception as e:
                index = (index + 1) % len(self.img_items)
                print(e)
        
        if self.relabel:
            camid = self.cam_dict[camid]
            
        if self.is_train:
            return {
                "image": img,
                "targets": pid,
                "camids": camid,
                "im_shape": im_shape,
                "scale_factor": scale_factor,
                "img_paths": img_path,
                "im_id": im_id,
                "id2imgname": self.id2imgname,
            }
        else:
            return {
                "image": img,
                "im_shape": im_shape,
                "scale_factor": scale_factor,
                "img_paths": img_path,
                "im_id": im_id,
                "id2imgname": self.id2imgname,
            }

    @property
    def num_classes(self):
        """get number of classes
        """
        return self._num_classes

    @property
    def num_cameras(self):
        """get number of cameras
        """
        return len(self.cams)


def build_hierachical_test_set(dataset_name=None, transforms=None, is_train=True, **kwargs):
    """
    build test_set for the tasks of Person, Veri and Sop
    """
    data = DATASET_REGISTRY.get(dataset_name)(root=_root, **kwargs)
    test_items = data.query + data.gallery
    
    #在末尾随机填充若干data.gallery数据使得test_items的长度能被world_size整除，以保证每个卡上的数据量均分；
    world_size = comm.get_world_size()
    
    if len(test_items)%world_size != 0:
        test_items += [random.choice(data.query + data.gallery) for _ in range(world_size - len(test_items)%world_size)]
    
    test_set = HierachicalCommDataset(test_items, transforms, relabel=False, \
        dataset_name=data.dataset_name, num_classes=kwargs.get('num_classes', 0), is_train=is_train)

    # Update query number
    test_set.num_query = len(data.query)
    
    #记录data.query和data.gallery的有效长度，在评估模块中，只取出来前有效长度的数据，丢弃末尾填充的数据
    test_set.num_valid_samples = len(data.query + data.gallery)
    
    return test_set


def build_hierachical_softmax_train_set(names=None, transforms=None, num_classes=1000, **kwargs):
    """build_hierachical_softmax_train_set for decathlon datasets
    """
    train_items = list()
    for d in names:
        data = DATASET_REGISTRY.get(d)(root=_root, **kwargs)
        # if comm.is_main_process():
        #     data.show_train()
        train_items.extend(data.train)

    train_set = HierachicalCommDataset(train_items, transforms, relabel=False, num_classes=num_classes)
    return train_set


def build_vehiclemulti_train_loader_lazy(
    train_set, sampler_config=None, total_batch_size=0, num_workers=0, dp_degree=None, alive_rank_list=None):
    """
    Build a dataloader for object re-identification with some default features.
    This interface is experimental.

    Returns:
        torch.utils.data.DataLoader: a dataloader.
    """
    if dp_degree is not None:
        if (alive_rank_list is not None) and (comm.get_rank() not in alive_rank_list):
            return {}
        dp_group = moe_group_utils.get_dp_group(dp_degree)
        moe_group = moe_group_utils.get_moe_group(dp_degree)
    else:
        dp_group = None
        moe_group = None
    sampler_name = sampler_config.get('sampler_name', 'TrainingSampler')
    num_instance = sampler_config.get('num_instance', 4)
    mini_batch_size = total_batch_size // comm.get_world_size(dp_group)

    if sampler_name == "TrainingSampler": 
        sampler = samplers.TrainingSampler(train_set, dp_group=dp_group, moe_group=moe_group)
        batch_sampler = paddle.io.BatchSampler(sampler=sampler, batch_size=mini_batch_size)
    elif sampler_name == 'ClassAwareSampler':
        batch_sampler = VehicleMultiTaskClassAwareSampler(dataset=train_set, batch_size=mini_batch_size)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    train_loader = paddle.io.DataLoader( #TODO make a distributed version
        dataset=train_set,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
        num_workers=num_workers,
        )

    return train_loader