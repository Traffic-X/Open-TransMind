# !/usr/bin/env python3
import os
import random
import logging
from collections.abc import Mapping

import numpy as np
import paddle

from utils import comm
from fastreid.data import samplers
from fastreid.data import CommDataset
from fastreid.data.data_utils import read_image
from fastreid.data.datasets import DATASET_REGISTRY
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


class MultiTaskDataLoader(object):
    """MultiTaskDataLoader
    """
    def __init__(self, task_loaders, cfg):
        super().__init__()
        self.task_loaders = task_loaders
        self.cfg = cfg

        self.task_iters = {}
        for name, loader in self.task_loaders.items():
            self.task_iters[name] = iter(loader)

    def __iter__(self):
        return self
        
    def __len__(self):
        # TODO: make it more general
        return len(list(self.task_iters.values())[0])

    def __next__(self):
        batch = {}

        if self.cfg.sample_mode == 'batch':
            for name, iter_ in self.task_iters.items():
                batch[name] = next(iter_)
        elif self.cfg.sample_mode == 'sample':
            name = random.choices(self.task_iters.keys(), self.cfg.sample_prob)[0]
            batch[name] = next(self.task_iters[name])
        else:
            raise NotImplementedError

        return batch

    # Signal for shutting down background thread
    def shutdown(self):
        """shutdown
        """
        for name, loader in self.task_loaders:
            loader.shutdown()


class MOEplusplusMultiTaskDataLoader(MultiTaskDataLoader):
    """MOEplusplusMultiTaskDataLoader
    """
    def __init__(self, task_loaders, cfg):
        globalrank2taskid = cfg['globalrank2taskid']
        rank = comm.get_rank()
        taskid = globalrank2taskid[str(rank)]
        taskname = list(task_loaders.keys())[taskid]
        # task_loader = {taskname: task_loaders[taskname]}
        print('rank {} has task_loader of {}'.format(rank, taskname))
        # super().__init__(task_loader, cfg)

        self.task_loaders = task_loaders
        self.cfg = cfg

        self.task_iters = {taskname: iter(task_loaders[taskname])}
        # for name, loader in self.task_loaders.items():
        #     self.task_iters[name] = iter(loader)


def build_reid_test_loader_lazy(test_set, test_batch_size, num_workers, dp_degree=None, alive_rank_list=None):
    """
    build reid test_loader for tasks of Person, Veri and Sop
    this test_loader only supports single gpu
    """
    if dp_degree is not None:
        if (alive_rank_list is not None) and (comm.get_rank() not in alive_rank_list):
            return {}
        dp_group = None
        moe_group = None
    else:
        dp_group = None
        moe_group = None
    mini_batch_size = test_batch_size // comm.get_world_size(dp_group)
    data_sampler = samplers.InferenceSampler(test_set, dp_group=dp_group, moe_group=moe_group)
    batch_sampler = paddle.io.BatchSampler(sampler=data_sampler, batch_size=mini_batch_size)
    test_loader = paddle.io.DataLoader(
        dataset=test_set,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
        num_workers=0,
        )
    return test_loader


def build_test_set(dataset_name=None, transforms=None, **kwargs):
    """
    build test_set for the tasks of Person, Veri and Sop
    """
    data = DATASET_REGISTRY.get(dataset_name)(root=_root, **kwargs)
    if comm.is_main_process():
        data.show_test()
    data.show_test()
    test_items = data.query + data.gallery
    #在末尾随机填充若干data.gallery数据使得test_items的长度能被world_size整除，以保证每个卡上的数据量均分；
    world_size = comm.get_world_size()
    if len(test_items)%world_size != 0:
        test_items += [random.choice(data.query + data.gallery) for _ in range(world_size - len(test_items)%world_size)]
    if dataset_name == 'SOP':
        test_set = CommDataset(test_items, transforms, relabel=True, dataset_name=data.dataset_name)
    else:
        test_set = CommDataset(test_items, transforms, relabel=False, dataset_name=data.dataset_name)

    # Update query number
    test_set.num_query = len(data.query)
    #记录data.query和data.gallery的有效长度，在评估模块中，只取出来前有效长度的数据，丢弃末尾填充的数据
    test_set.num_valid_samples = len(data.query + data.gallery)
    return test_set


def build_train_set(names=None, transforms=None, **kwargs):
    """
    build test_set for the tasks of Face, Person, Veri and Sop
    """
    train_items = list()
    for d in names:
        data = DATASET_REGISTRY.get(d)(root=_root, **kwargs)
        if comm.is_main_process():
            data.show_train()
        train_items.extend(data.train)

    train_set = CommDataset(train_items, transforms, relabel=True)
    return train_set



class TestFaceDataset(CommDataset):
    """TestFaceDataset
    """
    def __init__(self, img_items, labels, transforms=None, dataset_name=None):
        self.img_items = img_items
        self.labels = labels
        self.transforms = transforms
        self.dataset_name = dataset_name

    def __getitem__(self, index):
        img = read_image(self.img_items[index])
        if self.transforms is not None: img = self.transforms(img)
        return {"images": img,}


def build_face_test_set(dataset_name=None, transforms=None, **kwargs):
    """build_face_test_set
    """
    data = DATASET_REGISTRY.get(dataset_name)(root=_root, **kwargs)
    if comm.is_main_process():
        data.show_test()
    data.show_test()
    # style1
    # test_set = TestFaceDataset(data.img_paths, data.is_same, transforms, data.dataset_name)
    # test_set.num_query = len(data.img_paths)
    
    # style2
    # 在末尾随机填充若干data.gallery数据使得test_items的长度能被world_size整除，以保证每个卡上的数据量均分；
    test_items = data.img_paths
    test_is_same_list = data.is_same
    world_size = comm.get_world_size()
    if len(test_items)%world_size != 0:
        idx_list = list(range(len(data.img_paths)))
        random_idx_list = [random.choice(idx_list) for _ in range(world_size - len(test_items)%world_size)]
        test_items += [data.img_paths[idx] for idx in random_idx_list]
        # test_is_same_list += [data.is_same[idx] for idx in random_idx_list]
    test_set = TestFaceDataset(test_items, test_is_same_list, transforms, data.dataset_name)

    # Update query number
    test_set.num_query = len(data.img_paths)
    # 记录data.query和data.gallery的有效长度，在评估模块中，只取出来前有效长度的数据，丢弃末尾填充的数据
    test_set.num_valid_samples = len(data.img_paths)
    return test_set


def build_imagenet_train_set(names=None, transforms=None, **kwargs):
    """build_imagenet_train_set for decathlon datasets
    """
    train_items = list()
    for d in names:
        data = DATASET_REGISTRY.get(d)(root=_root, **kwargs)
        if comm.is_main_process():
            data.show_train()
        train_items.extend(data.train)

    train_set = CommDataset(train_items, transforms, relabel=False)
    return train_set


def build_reid_train_imagenet_loader_lazy(
        train_set, sampler_config=None, total_batch_size=0, num_workers=0, batch_ops=None
):
    """
    Build a dataloader for object re-identification with some default features.
    This interface is experimental.

    Returns:
        torch.utils.data.DataLoader: a dataloader.
    """
    def transform(data, ops=[]):
        """ transform """
        for op in ops:
            data = op(data)
        return data

    def mix_collate_fn(batched_inputs):
        """
        #batch[0] -> {
        #     "images": img,
        #     "targets": pid,
        #     "camids": camid,
        #     # "img_paths": img_path,
        # }
        """
        batch = []
        for batched_input in batched_inputs:
            batch.append((batched_input["images"], batched_input["targets"]))
        # batch = transform(batch, batch_ops)
        batch = batch_ops(batch)
        # batch each field
        slots = []
        for items in batch:
            for i, item in enumerate(items):
                if len(slots) < len(items):
                    slots.append([item])
                else:
                    slots[i].append(item)
        batch_data = [np.stack(slot, axis=0) for slot in slots]

        return {"images": batch_data[0], 
                "targets": batch_data[1], 
                "camids": np.array([int(batched_input["camids"]) for batched_input in batched_inputs])}

    sampler_name = sampler_config.get('sampler_name', 'TrainingSampler')
    num_instance = sampler_config.get('num_instance', 4)
    mini_batch_size = total_batch_size // comm.get_world_size()

    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    assert sampler_name == "TrainingSampler"
    sampler = samplers.TrainingSampler(train_set)

    batch_sampler = paddle.io.BatchSampler(sampler=sampler, batch_size=mini_batch_size, drop_last=True)
    train_loader = paddle.io.DataLoader( #TODO make a distributed version
        dataset=train_set,
        batch_sampler=batch_sampler,
        collate_fn=mix_collate_fn,
        num_workers=num_workers,
        )

    return train_loader


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
    data_set = DATASET_REGISTRY.get(dataset_name)(dataset_dir=os.path.join(dataset_dir, 'coco'), image_dir=image_dir,
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

def build_voc_set(dataset_name=None, transforms=[], dataset_dir=_root, image_dir='voc',
        anno_path='trainval.txt',
        data_fields=['image', 'gt_bbox', 'gt_class', 'difficult'], **kwargs):
    """
    build train_set for detection
    """
    import ppdet
    data_set = ppdet.data.voc.VOCDataSet(dataset_dir=os.path.join(dataset_dir, image_dir),
        anno_path=anno_path, label_list='label_list.txt', data_fields=data_fields)
    num_classes = 20    # hard code
    transforms = Compose(transforms, num_classes=num_classes)
    data_set.parse_dataset()
    data_set.set_transform(transforms)
    data_set.set_kwargs(**kwargs)
    data_set.dataset_name = dataset_name
    
    # to use coco evaluator, add catid2clsid and get_anno()
    data_set.catid2clsid = dict({catid: i for i, catid in 
                enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
                })
    def get_anno():
        return os.path.join(dataset_dir, image_dir, 'test.cocostyle.json')
    data_set.get_anno = get_anno
    print('{} has {} samples'.format(dataset_name, len(data_set))) #data_set.roidbs
    return data_set

def build_obj365det_set(dataset_name=None, transforms=[], dataset_dir=_root, image_dir='train2017',
        anno_path='annotations/instances_train2017.json',
        data_fields=['image', 'gt_bbox', 'gt_class', 'is_crowd'], **kwargs):
    """
    build train_set for detection
    """
    import ppdet
    data_set = ppdet.data.COCODataSet(
        dataset_dir=os.path.join(dataset_dir, 'Objects365'), 
        image_dir=image_dir,
        anno_path=anno_path, 
        data_fields=data_fields)
    num_classes = 365   # hard code
    transforms = Compose(transforms, num_classes=num_classes)
    data_set.parse_dataset()
    data_set.set_transform(transforms)
    data_set.set_kwargs(**kwargs)
    data_set.dataset_name = dataset_name
    print('{} has {} samples'.format(dataset_name, len(data_set))) #data_set.roidbs
    return data_set