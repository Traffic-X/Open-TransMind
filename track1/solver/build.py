"""solve/build.py
"""
from typing import Optional, Dict, List, Any, Set, Type
import re
import copy
import math

import paddle


def build_lr_optimizer_lazy(**kwargs):
    """build_lr_optimizer_lazy
    """
    model = kwargs['model']
    lr_multiplier = kwargs['lr_multiplier']
    optimizer_type = kwargs.get('optimizer_type', 'SGD') 
    momentum = kwargs.get('momentum', 0.9)
    weight_decay = kwargs.get('weight_decay', 1e-4)
    grad_clip_enabled = kwargs.get('grad_clip_enabled', True)
    grad_clip_norm = kwargs.get('grad_clip_norm', 5.0)
    apply_decay_param_fun = kwargs.get('apply_decay_param_fun', None)
    # grad_clip = paddle.nn.ClipGradByNorm(grad_clip_norm) if grad_clip_enabled else None
    grad_clip = paddle.nn.ClipGradByGlobalNorm(grad_clip_norm) if grad_clip_enabled else None

    if optimizer_type == 'SGD':
        return  paddle.optimizer.Momentum(
                learning_rate=lr_multiplier,
                momentum=momentum,
                parameters=model.parameters(),
                weight_decay=weight_decay,
                grad_clip=grad_clip,
            )
    elif optimizer_type == 'Adam':
        return paddle.optimizer.Adam(
                learning_rate=lr_multiplier, 
                beta1=0.9,
                beta2=0.999, 
                epsilon=1e-08, 
                parameters=model.parameters(), 
                weight_decay=weight_decay, 
                grad_clip=grad_clip, 
                name=None, 
                lazy_mode=False,
            )
    elif optimizer_type == 'AdamW':
        return paddle.optimizer.AdamW(
                learning_rate=lr_multiplier, 
                beta1=0.9, beta2=0.999, 
                epsilon=1e-08, 
                parameters=model.parameters(), 
                weight_decay=weight_decay, 
                lr_ratio=None, 
                apply_decay_param_fun=apply_decay_param_fun, 
                grad_clip=grad_clip, 
                lazy_mode=False, 
                multi_precision=False, 
                name=None,
            )
    else:
        raise ValueError()


def build_lr_scheduler_lazy(**kwargs):
    """build_lr_scheduler_lazy
    """
    max_iters = kwargs['max_iters']
    sched =  kwargs['sched']
    base_lr = kwargs['base_lr']
    warmup_iters = kwargs.get('warmup_iters', 0)
    warmup_method = kwargs.get('warmup_method', 'linear')
    eta_min = kwargs.get('eta_min', 1e-8)
    solver_steps = kwargs.get('solver_steps', [20000])
    solver_gamma = kwargs.get('solver_gamma', 0.1)

    power = kwargs.get('power', 0.9)
    warmup_start_lr = kwargs.get('warmup_start_lr', 1.0e-5)
    end_lr = kwargs.get('end_lr', 0.0)

    if warmup_method == 'linear' and sched == 'CosineAnnealingLR': 
        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            paddle.optimizer.lr.CosineAnnealingDecay(base_lr, max_iters, eta_min), 
            warmup_iters, 
            0., 
            base_lr)
    elif warmup_iters == 0 and sched == 'PiecewiseDecay': 
        lr_steps = [pow(solver_gamma, i) * base_lr for i in range(len(solver_steps) + 1)]
        lr_scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=solver_steps, values=lr_steps)
    elif warmup_iters > 0 and sched == 'PiecewiseDecay': 
        lr_steps = [pow(solver_gamma, i) * base_lr for i in range(len(solver_steps) + 1)]
        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            paddle.optimizer.lr.PiecewiseDecay(boundaries=solver_steps, values=lr_steps),
            warmup_iters,
            eta_min,
            base_lr
        )
    elif warmup_iters > 0 and sched == 'PolynomialDecay':   # add for segmentation
        decay_steps = max_iters - warmup_iters
        lr_sche = paddle.optimizer.lr.PolynomialDecay(base_lr, power=power, decay_steps=decay_steps, end_lr=0)
        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
                learning_rate=lr_sche,
                warmup_steps=warmup_iters,
                start_lr=warmup_start_lr,
                end_lr=base_lr)
    else:
        raise ValueError("Unknown warmup and sched method : {} and {}".format(warmup_method, sched))
    return lr_scheduler
