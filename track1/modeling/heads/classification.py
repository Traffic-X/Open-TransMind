# !/usr/bin/env python3
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from collections import OrderedDict
import pycocotools.mask as mask_util
from modeling.initializer import linear_init_, constant_
from modeling.losses import triplet_loss, cross_entropy_loss, log_accuracy
from paddle.nn.initializer import TruncatedNormal, Constant, Normal
from paddle import ParamAttr
from layers import any_softmax
import random

trunc_normal_ = TruncatedNormal(std=.02)
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)
normal_ = Normal
BIAS_LR_FACTOR=2.0

class ClassificationNeck(nn.Layer):
    """d
    """
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, x):
        """d
        """
        if isinstance(x, (list, tuple)):
            return x[-1]
        else:
            return x

class GlobalAvgPool(nn.AdaptiveAvgPool2D):
    """
    GlobalAvgPool
    """
    def __init__(self, output_size=1, *args, **kwargs):
        """Init
        """
        super().__init__(output_size)

class ClassificationHead(nn.Layer):
    """d
    """
    def __init__(self, feat_dim, num_classes, neck, scale=1,
            margin=0, load_head=False, pretrain_path='', **loss_kwargs,
            ):
        super().__init__()
        self.neck = neck
        self.pool_layer = GlobalAvgPool()
        assert num_classes > 0
        self.linear = paddle.nn.Linear(feat_dim, num_classes, 
            bias_attr=ParamAttr(learning_rate=0.1 * BIAS_LR_FACTOR),
            weight_attr=ParamAttr(learning_rate=0.1)
            )
        self.cls_layer = getattr(any_softmax, "Linear")(num_classes, scale, margin)
        if load_head:
            # pretrain_path
            state_dict = paddle.load(pretrain_path)
            print("Loading Head from {}".format(pretrain_path))
            if 'model' in state_dict:
                state_dict = state_dict.pop('model')
            if 'state_dict' in state_dict:
                state_dict = state_dict.pop('state_dict')
            state_dict_new = OrderedDict()
            for k, v in state_dict.items():
                if 'head' in k:
                    k_new = k[5:]
                    state_dict_new[k_new] = state_dict[k]
            self.linear.set_state_dict(state_dict_new)
        
        self.ce_kwargs = loss_kwargs.get('ce', {})
    
    def forward(self, body_feats, inputs):
        """d
        """
        neck_feat = self.neck(body_feats)
        feat = self.pool_layer(neck_feat)
        feat = feat[:, :, 0, 0]
        targets = inputs["targets"]
        if self.training:
            loss_dict = {}
            logits = self.linear(feat)
            pred_class_logits = logits * self.cls_layer.s
            cls_outputs = self.cls_layer(logits, targets)
            # Log prediction accuracy
            # acc = log_accuracy(pred_class_logits, gt_labels)
            ce_prob = self.ce_kwargs.get('prob', 1.0)
            if random.random() < ce_prob:
                loss_exist = 1.0
            else:
                loss_exist = 0.0
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs,
                targets,
                self.ce_kwargs.get('eps', 0.0),
                self.ce_kwargs.get('alpha', 0.2)
            ) * self.ce_kwargs.get('scale', 1.0) * loss_exist
            return loss_dict
        else:
            return self.linear(feat)
