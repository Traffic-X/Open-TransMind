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
import glob

from data.datasets.seg_dataset import Dataset
from paddleseg.cvlibs import manager
from data.transforms.seg_transforms import Compose
from fastreid.data.datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class BDD100K(Dataset):
    """
    Args:
        transforms (list): Transforms for image.
        dataset_root (str): Cityscapes dataset directory.
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        edge (bool, optional): Whether to compute edge while training. Default: False
    """
    NUM_CLASSES = 19
    dataset_name = 'BDD100K'
    
    def __init__(self, transforms, dataset_root, mode='train', edge=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = list()
        mode = mode.lower()
        self.mode = mode
        # self.num_classes = self.NUM_CLASSES
        self.ignore_index = 255
        self.edge = edge

        if mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        img_dir = os.path.join(self.dataset_root, 'images')
        label_dir = os.path.join(self.dataset_root, 'label')
        if self.dataset_root is None or not os.path.isdir(
                self.dataset_root) or not os.path.isdir(
                    img_dir) or not os.path.isdir(label_dir):
            raise ValueError(
                "The dataset is not Found or the folder structure is nonconfoumance."
            )

        label_files = sorted(
            glob.glob(
                os.path.join(label_dir, mode, '*.png')))
        img_files = sorted(
            glob.glob(os.path.join(img_dir, mode, '*.jpg')))

        self.file_list = [
            [img_path, label_path]
            for img_path, label_path in zip(img_files, label_files)
        ]


@DATASET_REGISTRY.register()
class InferDataset(Dataset):
    """
    Infer Dataset
    """
    NUM_CLASSES = 19
    dataset_name = 'InferDataset'

    def __init__(self,
                 mode,
                 dataset_root,
                 transforms,
                 num_classes=19,
                 img_channels=3,
                 ignore_index=255,
                 edge=False):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms, img_channels=img_channels)
        self.file_list = list()
        self.mode = mode.lower()
        self.num_classes = 19
        self.img_channels = img_channels
        self.ignore_index = ignore_index
        self.edge = edge

        if self.mode not in ['train', 'val', 'test']:
            raise ValueError(
                "mode should be 'train', 'val' or 'test', but got {}.".format(
                    self.mode))

        img_dir = os.path.join(self.dataset_root, 'images')
        if self.dataset_root is None or not os.path.isdir(
                self.dataset_root) or not os.path.isdir(img_dir):
            raise ValueError(
                "The dataset is not Found or the folder structure is nonconfoumance."
            )

        img_files = sorted(
            glob.glob(os.path.join(img_dir, self.mode, '*.jpg')))

        self.file_list = [(idx, img_path) for idx, img_path in enumerate(img_files)]
        self.id2path = {}
        for idx, img_path in enumerate(img_files):
            self.id2path[idx] = os.path.basename(img_path)

    def __getitem__(self, idx):
        data = {}
        data['trans_info'] = []
        im_id, image_path = self.file_list[idx]
        data['image'] = image_path
        data['im_path'] = image_path
        data['im_id'] = im_id
        data['id2path'] = [self.id2path]
        # If key in gt_fields, the data[key] have transforms synchronous.
        data['gt_fields'] = []
        if self.mode == 'test':
            data = self.transforms(data)
        return data

    def __len__(self):
        return len(self.file_list)