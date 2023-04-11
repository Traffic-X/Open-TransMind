# !/usr/bin/env python3
"""Fine-Grained Visual Classification
"""
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

import os
import numpy as np


@DATASET_REGISTRY.register()
class FGVCDataset(ImageDataset):
    dataset_name = "FGVCDataset"

    def __init__(self, root='', **kwargs):
        self.root = root
        self.train_dataset_dir = kwargs['train_dataset_dir']
        self.test_dataset_dir = kwargs['test_dataset_dir']
        self.train_label = kwargs['train_label']
        self.test_label = kwargs['test_label']
        self.dict_label = self.init_dict_label(f'{self.train_label}')
        train = self.process_dir(self.train_dataset_dir, self.train_label)
        query = self.process_dir(self.test_dataset_dir, self.test_label)
        gallery = []
        super().__init__(train, query, gallery, **kwargs)

    def init_dict_label(self, label_dir):
        dict_label = {}
        count = 0
        with open(label_dir, 'r') as f:
            list_line = f.readlines()
        for line in list_line:
            line = line.strip()
            path, class_id = line.split()
            if class_id not in dict_label.keys():
                dict_label[class_id] = count
                count += 1
        return dict_label

    def process_dir(self, img_dir, label_dir):
        data = []
        with open(label_dir, 'r') as f:
            list_line = f.readlines()
        im_id = 0
        for line in list_line:
            line = line.strip()
            path, class_id = line.split()

            img_name = os.path.join(img_dir, path)
            data.append([img_name, int(class_id), '0', im_id])
            im_id += 1
        return data

# 构建测试Dataset
@DATASET_REGISTRY.register()
class FGVCInferDataset():
    dataset_name = "FGVCInferDataset"

    def __init__(self, root=None, **kwargs):
        self.root = root
        self.test_dataset_dir = kwargs['test_dataset_dir']
        self.query = self.process_dir(self.test_dataset_dir)
        self.gallery = []

    def process_dir(self, img_dir):
        data = []   
        files = os.listdir(img_dir)
        im_id = 0
        for line in files:
            img_path = os.path.join(img_dir, line)
            data.append([img_path, 0, '0', im_id])
            im_id += 1
        return data