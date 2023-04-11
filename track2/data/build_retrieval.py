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
from data.datasets.retrieval_dataset import *


def build_retrieval_dataset(dataset_name=None, transforms=None, dataroot=None, **kwargs):
    """
    Build Retrieval Datasets
    """
    data_set = DATASET_REGISTRY.get(dataset_name)(dataroot=dataroot, transforms=transforms)
    print("data_set:", data_set)
    print('{} has {} samples'.format(dataset_name, data_set.__len__()))  # data_set.roidbs
    return data_set


def build_retrieval_trainloader(data_set, is_train=True, total_batch_size=0, \
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



def build_retrieval_test_dataset(dataset_name=None, transforms=[], dataset_root=None, 
        mode='val', **kwargs):
    data_set = DATASET_REGISTRY.get(dataset_name)(dataset_root=dataset_root, transforms=transforms, mode=mode, **kwargs)

    print('{} has {} samples'.format(dataset_name, len(data_set.file_list)))  # data_set.roidbs
    return data_set


if __name__ == '__main__':
    from data.transforms.build import build_transforms_lazy

    retrieval=build_retrieval_trainloader(
        data_set=build_retrieval_dataset(
                dataset_name="RetrievalDataset",
                dataroot='/root/paddlejob/workspace/env_run/datasets/textimage/person_car',
                transforms=build_transforms_lazy(
                    is_train=True,
                    size_train=[224, 224],
                    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                ),),
        total_batch_size=16, 
        worker_num=4, 
        drop_last=True, 
        shuffle=True,
        is_train=True,
    )
    for i in retrieval:
        print(i)