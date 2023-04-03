# !/usr/bin/env python3

import os
import random
import logging
from collections.abc import Mapping

import numpy as np
import paddle

from utils import comm
from fastreid.data import samplers
from fastreid.data.datasets import DATASET_REGISTRY
from tools import moe_group_utils
from data.transforms import detection_ops
from data.transforms.detection_ops import Compose, BatchCompose


_root = os.getenv("FASTREID_DATASETS", "datasets")


def fast_batch_collator(batched_inputs):
    """
    A simple batch collator for most common reid tasks.
    There is no need of transforming data to GPU in fast_batch_collator
    """
    elem = batched_inputs[0]
    if isinstance(elem, np.ndarray):
        # return paddle.to_tensor(np.concatenate([ np.expand_dims(elem, axis=0) for elem in batched_inputs], axis=0))
        return np.concatenate([np.expand_dims(elem, axis=0) for elem in batched_inputs], axis=0)

    elif isinstance(elem, Mapping):
        return {key: fast_batch_collator([d[key] for d in batched_inputs]) for key in elem}
    elif isinstance(elem, float):
        # return paddle.to_tensor(batched_inputs, dtype=paddle.float64)
        return np.array(batched_inputs, dtype=np.float64) 
    elif isinstance(elem, int):
        #return paddle.to_tensor(batched_inputs)
        return np.array(batched_inputs) 
    elif isinstance(elem, str):
        return batched_inputs


def build_cocodet_test_loader_lazy(data_set, total_batch_size=0, num_workers=0, is_train=False,
        batch_transforms=[], shuffle=True, drop_last=True, collate_batch=True):
    """
    Build a dataloader for coco detection with some default features.

    Returns:
        paddle.io.DataLoader: a dataloader.
    """
    assert is_train == False
    num_classes = 80    # hard code
    batch_transforms = BatchCompose(batch_transforms, num_classes, collate_batch)
    batch_sampler = paddle.io.BatchSampler(dataset=data_set, batch_size=total_batch_size, drop_last=False, shuffle=False)
    data_loader = paddle.io.DataLoader(
        dataset=data_set,
        batch_sampler=batch_sampler,
        collate_fn=batch_transforms,
        num_workers=num_workers,
        return_list=False,
        use_shared_memory=False,
    )
    return data_loader


def build_cocodet_loader_lazy(data_set, total_batch_size=0, num_workers=0, is_train=False,
        batch_transforms=[], shuffle=True, drop_last=True, collate_batch=True):
    """
    Build a dataloader for coco detection with some default features.

    Returns:
        paddle.io.DataLoader: a dataloader.
    """
    mini_batch_size = total_batch_size // comm.get_world_size()
    num_classes = 80    # hard code
    batch_transforms = BatchCompose(batch_transforms, num_classes, collate_batch)
    
    if is_train:
        # 无限流
        sampler = samplers.TrainingSampler(data_set, shuffle=shuffle)
    else:
        # 有序分布流
        sampler = samplers.OrderInferenceSampler(data_set, mini_batch_size)
    
    batch_sampler = paddle.io.BatchSampler(sampler=sampler, batch_size=mini_batch_size)
    data_loader = paddle.io.DataLoader(
        dataset=data_set,
        batch_sampler=batch_sampler,
        collate_fn=batch_transforms,
        num_workers=num_workers,
        return_list=False,
        use_shared_memory=False,
    )
    
    return data_loader


def build_cocodet_set(dataset_name=None, transforms=[], dataset_dir=_root, image_dir='train2017',
        anno_path='annotations/instances_train2017.json',
        is_padding=False,
        data_fields=['image', 'gt_bbox', 'gt_class', 'is_crowd'], **kwargs):
    """
    build train_set for detection
    """
    data_set = DATASET_REGISTRY.get(dataset_name)(dataset_dir=os.path.join(dataset_dir), image_dir=image_dir,
        anno_path=anno_path, data_fields=data_fields)
    num_classes = 80    # hard code
    transforms = Compose(transforms, num_classes=num_classes)
    data_set.parse_dataset()
    data_set.set_transform(transforms)
    data_set.set_kwargs(**kwargs)
    print('{} has {} samples'.format(dataset_name, len(data_set))) #data_set.roidbs
    if is_padding:
        # record sample number
        data_set.num_valid_samples = len(data_set)
        # 在末尾随机填充若干data.gallery数据使得test_items的长度能被world_size整除，以保证每个卡上的数据量均分；
        test_items = data_set.roidbs
        world_size = comm.get_world_size()
        if len(test_items)%world_size != 0:
            idx_list = list(range(len(test_items)))
            random_idx_list = [random.choice(idx_list) for _ in range(world_size - len(test_items)%world_size)]
            test_items += [test_items[idx] for idx in random_idx_list]
        data_set.roidbs = test_items
        print('{} has {} samples after padding'.format(dataset_name, len(data_set))) #data_set.roidbs
    return data_set