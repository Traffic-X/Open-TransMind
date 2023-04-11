# !/usr/bin/env python3
"""Fine-Grained Visual Classification
"""
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset
from ..util.tokenizer import tokenize
# from .preprocess import build_transforms
from fastreid.data.data_utils import read_image

import os
import PIL
import random
import paddle
import numpy as np



@DATASET_REGISTRY.register()
class RetrievalDataset(paddle.io.Dataset):
    dataset_name = "RetrievalDataset"

    def __init__(self,
                 dataroot,
                 transforms=None,
                 shuffle=False):
        """Create a text image dataset from a directory with congruent text and image names.

        Args:
            folder (str): Folder containing images and text files matched by their paths' respective "stem"
            image_size (int, optional): The size of outputted images. Defaults to 224.
            resize_ratio (float, optional): Minimum percentage of image contained by resize. Defaults to 0.75.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
        """
        super(RetrievalDataset, self).__init__()
        self.transforms = transforms
        # if transforms is not None:
        #     self.image_transform = build_transforms(transforms)
        self.shuffle = shuffle
        self.dataroot = dataroot # /root/paddlejob/workspace/env_run/datasets/textimage/car/trainset

        self.data = []
        images = os.listdir(os.path.join(self.dataroot, 'train_images/'))
        txt = os.path.join(self.dataroot, 'train_label.txt')
        f = open(txt, 'r')
        for line in f.readlines():
            line = line.strip()
            items = line.split('$')
            name, _, text = items
            image_path = os.path.join(self.dataroot, 'train_images/', name)
            self.data.append([image_path, text, name])

    def __len__(self):
        return len(self.data)

    def random_sample(self):
        return self.__getitem__(random.randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        image_file, text, name = self.data[ind]

        # descriptions = text_file.read_text().split('\n')
        # descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        # try:
        #     description = choice(descriptions)
        # except IndexError as zero_captions_in_file_ex:
        #     print(f"An exception occurred trying to load file {text_file}.")
        #     print(f"Skipping index {ind}")
        #     return self.skip_sample(ind)
        # f = open(text_file, 'r')
        # for line in f.readlines():
        description = text
        # print('load', name, description)

        tokenized_text = tokenize(description)[0]
        # print(tokenized_text)
        # img = read_image(image_file)
        # img = self.transforms(img)
        try:
            # image_tensor = self.image_transform(PIL.Image.open(image_file))
            img = read_image(image_file, "RGB")
            img = self.transforms(img)
        except:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Success
        return {'image': img, 'text': tokenized_text}
