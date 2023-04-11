from .common import train
import os
from omegaconf import OmegaConf
from collections import OrderedDict
from detectron2.config import LazyCall as L
from data.build import MultiTaskDataLoader
from modeling.meta_arch.multitask_v2 import MultiTaskBatchFuse

# retrieval
from data.transforms.build import build_transforms_lazy
from data.build_retrieval import  build_retrieval_dataset, \
    build_retrieval_trainloader, build_retrieval_test_dataset

from solver.build import build_lr_optimizer_lazy, build_lr_scheduler_lazy
    

dataloader=OmegaConf.create()
_root = "datasets"


dataloader.train=L(MultiTaskDataLoader)(
    cfg=dict(sample_mode='batch',),
    task_loaders=L(OrderedDict)(
        retrieval=L(build_retrieval_trainloader)(
            data_set=L(build_retrieval_dataset)(
                    dataset_name="RetrievalDataset",
                    dataroot=_root + '/train',
                    transforms=L(build_transforms_lazy)(
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
        ),
    ),
)


from modeling.backbones.vit_retrieval import CLIP
from modeling.heads.retrieval_head import RetrievalHead


backbone=L(CLIP)(
    embed_dim=512,
    image_resolution=224,
    vision_layers=12,
    vision_width=768,
    vision_patch_size=32,
    context_length=77,
    vocab_size=49408,
    transformer_width=512,
    transformer_heads=8,
    transformer_layers=12,
    qkv_bias=True,
    pre_norm=True,
    proj=True,
    patch_bias=False
)

model=L(MultiTaskBatchFuse)(
    backbone=backbone,
    heads=L(OrderedDict)(

        retrieval=L(RetrievalHead)(),

    ),
    pixel_mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    pixel_std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
)


optimizer = L(build_lr_optimizer_lazy)(
    optimizer_type='AdamW',
    base_lr=1e-4,
    weight_decay=1e-4,
    grad_clip_enabled=True,
    grad_clip_norm=0.1,
    apply_decay_param_fun=None,
    lr_multiplier=L(build_lr_scheduler_lazy)(
        max_iters=900000,
        warmup_iters=0,
        solver_steps=[720000],
        solver_gamma=0.1,
        base_lr=1e-4,
        sched='PiecewiseDecay',
    ),
)

train.amp.enabled = False

# data settings
sample_num = 136117     #训练集样本量
epochs=20
dataloader.train.task_loaders.retrieval.total_batch_size = 128 * 8 

iters_per_epoch = sample_num // dataloader.train.task_loaders.retrieval.total_batch_size

max_iters = iters_per_epoch * epochs

# optimizer
optimizer.lr_multiplier.max_iters = max_iters
optimizer.base_lr = optimizer.lr_multiplier.learning_rate = 1e-4
optimizer.lr_multiplier.solver_steps = [int(max_iters * 0.8)]


train.max_iter = max_iters
train.checkpointer.period = int(iters_per_epoch * 10)
train.checkpointer.max_to_keep=10    # 只保存最新的10个模型


train.output_dir = 'outputs/vitbase_retrieval'

# resume settings (remember last_checkpoint and --resume)
train.log_period = 20

train.init_checkpoint = 'pretrained/vitbase_clip.pdparams' # 导入CLIP预训练模型