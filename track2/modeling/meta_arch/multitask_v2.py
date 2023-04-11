# !/usr/bin/env python3
# Copyright (c) 2022  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import random

import paddle
from paddle import nn

from modeling.losses import triplet_loss, cross_entropy_loss, log_accuracy

class MultiTaskBatchFuse(nn.Layer):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    def __init__(
            self,
            backbone,
            heads,
            pixel_mean,
            pixel_std,
            task_loss_kwargs=None,
            task2head_mapping=None,
            **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
        """
        super().__init__()
        # backbone
        self.backbone = backbone

        # head
        # use nn.LayerDict to ensure head modules are properly registered
        self.heads = nn.LayerDict(heads)

        if task2head_mapping is None:
            task2head_mapping = {}
            for key in self.heads:
                task2head_mapping[key] = key
        self.task2head_mapping = task2head_mapping

        self.task_loss_kwargs = task_loss_kwargs

        self.register_buffer('pixel_mean', paddle.to_tensor(list(pixel_mean)).reshape((1, -1, 1, 1)), False)
        self.register_buffer('pixel_std', paddle.to_tensor(list(pixel_std)).reshape((1, -1, 1, 1)), False)

    @property
    def device(self):
        """
        Get device information
        """
        return self.pixel_mean.device

    def forward(self, task_batched_inputs):
        """
        NOTE: this forward function only supports `self.training is False`
        """
        # fuse batch
        img_list = []
        task_data_idx = {}
        start = 0
        # for task_name, batched_inputs in task_batched_inputs.items():
        #     images = self.preprocess_image(batched_inputs)
        #     img_list.append(images)

        #     end = start + images.shape[0]
        #     task_data_idx[task_name] = (start, end)
        #     start = end
        # all_imgs = paddle.concat(img_list, axis=0)
        # all_features = self.backbone(all_imgs)

        # assert not self.training
        losses = {}
        outputs = {}
        for task_name, batched_inputs in task_batched_inputs.items():
            # start, end = task_data_idx[task_name]
            # features = all_features[start:end, ...]
            features = self.backbone(self.preprocess_image(batched_inputs))

            if self.training:
                # assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
                # targets = batched_inputs["targets"]

                # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
                # may be larger than that in the original dataset, so the circle/arcface will
                # throw an error. We just set all the targets to 0 to avoid this problem.
                # if targets.sum() < 0: targets.zero_()

                task_outputs = self.heads[self.task2head_mapping[task_name]](features, batched_inputs)
                losses.update(**{task_name + "_" + key: val for key, val in task_outputs.items()})
            else:
                task_outputs = self.heads[self.task2head_mapping[task_name]](features, batched_inputs)
                outputs[task_name] = task_outputs

        if self.training:
            return losses
        else:
            return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            if 'image' in batched_inputs:
                images = batched_inputs['image']
            if 'images' in batched_inputs:
                images = batched_inputs['images']
            if 'text' in batched_inputs:
                texts = batched_inputs['text']
        elif isinstance(batched_inputs, paddle.Tensor):
            images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or Tensor, but get {}".format(type(batched_inputs)))

        # images.sub_(self.pixel_mean).div_(self.pixel_std)
        return {'image': images, 'text': texts}
