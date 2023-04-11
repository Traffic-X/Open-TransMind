#!/usr/bin/env python3
# Copyright (c) Baidu, Inc. and its affiliates.
"""
This training script is mainly constructed on train_net.py.
Additionally, this script is specialized for the training of supernet.
Moreover, this script adds a function of self-distillation.
If specifing `teacher_model_path` in the given config file, teacher model will
be built, otherwise teacher model is None.
"""
import logging
import os.path
import sys
import os

import paddle
import numpy as np
import json
sys.path.append('.')

SEED = os.getenv("SEED", "0")
paddle.seed(42)
# np.random.seed(int(SEED))

from utils.events import CommonMetricSacredWriter
from engine.hooks import LRScheduler
from utils.config import auto_adjust_cfg
from fastreid.utils.checkpoint import Checkpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
)
from evaluation import print_csv_format
from evaluation.evaluator import inference_on_dataset
from evaluation.seg_evaluator import seg_inference_on_dataset, seg_inference_on_test_dataset
from utils import comm

logger = logging.getLogger("ufo")


def do_test(cfg, model, _run=None, subnet_mode="largest"):
    if "evaluator" in cfg.dataloader:
        dataloaders = instantiate(cfg.dataloader.test)
        pred_rets = {}
        for idx, (dataloader, evaluator_cfg) in enumerate(zip(dataloaders, cfg.dataloader.evaluator)):
            task_name = '.'.join(list(dataloader.task_loaders.keys()))
            dataset_name = dataloader.task_loaders[task_name].dataset.dataset_name
            if (hasattr(cfg.train, 'selected_task_names')) and (task_name not in cfg.train.selected_task_names):
                continue
            print('=' * 10, dataset_name, '=' * 10)
            # recognition
            if hasattr(list(dataloader.task_loaders.values())[0].dataset, 'num_query'):
                evaluator_cfg.num_query = list(dataloader.task_loaders.values())[0].dataset.num_query
                evaluator_cfg.num_valid_samples = list(dataloader.task_loaders.values())[0].dataset.num_valid_samples
                evaluator_cfg.labels = list(dataloader.task_loaders.values())[0].dataset.labels
                evaluator = instantiate(evaluator_cfg)
                ret = inference_on_dataset(model, dataloader, evaluator)
            # segmentation
            elif dataset_name in['Cityscapes', 'BDD100K', 'InferDataset']:
                evaluator = instantiate(evaluator_cfg)
                print("seg_inference_on_test_dataset")
                ret = seg_inference_on_test_dataset(model, dataloader, evaluator)
            # detection
            else:
                evaluator_cfg.anno_file = list(dataloader.task_loaders.values())[0].dataset.get_anno()
                evaluator_cfg.clsid2catid = {v: k for k, v in list(dataloader.task_loaders.values())[0].dataset.catid2clsid.items()}
                evaluator = instantiate(evaluator_cfg)
                ret = inference_on_dataset(model, dataloader, evaluator)
            if comm.is_main_process():
                pred_rets.update(**ret)

        if comm.is_main_process():
            save_path = cfg.train.output_dir
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, 'pred_results.json')
            with open(save_path, 'w') as f:
                json.dump(pred_rets, f)
                logger.info(f'Pred results are saved to {save_path}')
def main(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        print('rank is {} , world_size is {}, gpu is {} '.format(args.rank, args.world_size, args.gpu))

    paddle.set_device('gpu')
    rank = paddle.distributed.get_rank()
    print('rank is {}, world size is {}'.format(rank, paddle.distributed.get_world_size()))
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)
    Checkpointer(model).load(cfg.train.init_checkpoint)
    do_test(cfg, model)
    
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    
    main(args)
