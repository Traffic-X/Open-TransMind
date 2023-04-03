# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import paddle
import paddleseg

from utils import comm
import numpy as np
from fastreid.data import samplers
from fastreid.data.datasets import DATASET_REGISTRY
from data.datasets.cityscapes_datasets import *
from data.datasets.bdd100k_datasets import *


def build_segmentation_dataset(dataset_name=None, transforms=[], dataset_root=None, 
        mode='train', **kwargs):
    """
    Build Cityscapes Datasets
    """
    data_set = DATASET_REGISTRY.get(dataset_name)(dataset_root=dataset_root, transforms=transforms, mode=mode, **kwargs)
    print("data_set:", data_set)
    print('{} has {} samples'.format(dataset_name, len(data_set.file_list)))  # data_set.roidbs
    return data_set


def build_segmentation_trainloader(data_set, is_train=True, total_batch_size=0, \
        worker_num=0, drop_last=True, **kwargs):
    """
    Build a dataloader for Cityscapse segmentation.
    Returns:
        paddle.io.DataLoader: a dataloader.
    """
    
    mini_batch_size = total_batch_size // comm.get_world_size()

    if is_train:
        # 无限流
        sampler = samplers.TrainingSampler(data_set)
        batch_sampler = paddle.io.BatchSampler(sampler=sampler, batch_size=mini_batch_size)  
        worker_init_fn = np.random.seed(random.randint(0, 100000))   
    else:
        # 有序分布流
        _batch_sampler = samplers.OrderInferenceSampler(data_set, mini_batch_size)
        batch_sampler = paddle.io.BatchSampler(sampler=_batch_sampler, batch_size=mini_batch_size)
        # batch_sampler = paddle.io.BatchSampler(dataset=data_set, batch_size=mini_batch_size, \
        #     shuffle=False, drop_last=drop_last)
        worker_init_fn=None
    
    dataloader = paddle.io.DataLoader(
        dataset=data_set,
        batch_sampler=batch_sampler,
        num_workers=worker_num,
        return_list=True,
        worker_init_fn=worker_init_fn)
    return dataloader



def build_segementation_test_dataset(dataset_name=None, transforms=[], dataset_root=None, 
        mode='val', is_padding=True, **kwargs):
    data_set = DATASET_REGISTRY.get(dataset_name)(dataset_root=dataset_root, transforms=transforms, mode=mode, **kwargs)

    print('{} has {} samples'.format(dataset_name, len(data_set.file_list)))  # data_set.roidbs
    if is_padding:
        # record sample number
        data_set.num_valid_samples = len(data_set)
        # 在末尾随机填充若干data.gallery数据使得test_items的长度能被world_size整除，以保证每个卡上的数据量均分；
        test_items = data_set.file_list
        world_size = comm.get_world_size()
        if len(test_items)%world_size != 0:
            idx_list = list(range(len(test_items)))
            random_idx_list = [random.choice(idx_list) for _ in range(world_size - len(test_items)%world_size)]
            test_items += [test_items[idx] for idx in random_idx_list]
        data_set.file_list = test_items
        print('{} has {} samples after padding'.format(dataset_name, len(data_set))) #data_set.roidbs
    
    return data_set

