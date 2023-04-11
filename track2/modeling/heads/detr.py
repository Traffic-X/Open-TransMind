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


class DETR(nn.Layer):
    """d
    """
    def __init__(self, transformer, detr_head, post_process, exclude_post_process=False):
        super().__init__()

        self.transformer = transformer
        self.detr_head = detr_head
        self.post_process = post_process
        self.exclude_post_process = exclude_post_process
        self.start = 0

    def forward(self, body_feats, inputs):
        """d"""
        # body_feats = self.backbone(inputs)
        # if self.start == 0:
        #     self.transformer.set_state_dict(paddle.load('self.transformer.pddet.pdparams'))
        #     self.start += 1
        if isinstance(body_feats, list):
            body_feats = body_feats[0]
        else:
            body_feats = body_feats
        # Transformer
        if 'gt_bbox' in inputs:
            #hard code
            gt_bbox = [paddle.cast(inputs['gt_bbox'][i], 'float32') for i in range(len(inputs['gt_bbox']))]
            inputs['gt_bbox'] = gt_bbox
        pad_mask = inputs['pad_mask'] if self.training else None
        out_transformer = self.transformer(body_feats, pad_mask, inputs)
        #
        # out_transformer = paddle.load('out_transformer.pddata')
        # body_feats = paddle.load('body_feats.pddata')
        # inputs = paddle.load('inputs.pddata')
        # d = self.detr_head(out_transformer, body_feats, inputs)
        # self.transformer.eval()
        # with paddle.no_grad():
        #     e1 = self.transformer(body_feats, inputs['pad_mask'], inputs)
        #     e2 = self.transformer(body_feats, inputs['pad_mask'], inputs)
        # paddle.save(self.transformer.state_dict(), 'self.transformer.pdparams')
        # with paddle.no_grad():
        #     e1 = self.transformer._get_encoder_input(body_feats, inputs['pad_mask'])
        #     e2 = self.transformer._get_encoder_input(body_feats, inputs['pad_mask'])
        # with paddle.no_grad():
        #     m1 = self.transformer.encoder(*e1)
        #     m2 = self.transformer.encoder(*e2)
        # DETR Head
        if self.training:
            losses = self.detr_head(out_transformer, body_feats, inputs)
            new_losses = {}
            new_losses.update({
                'loss':
            paddle.add_n([v for k, v in losses.items() if 'log' not in k])
            })
            return new_losses
        else:
            preds = self.detr_head(out_transformer, body_feats)
            if self.exclude_post_process:
                bboxes, logits, masks = preds
                bbox_pred, bbox_num = bboxes, logits
            else:
                bbox, bbox_num = self.post_process(
                    preds, inputs['im_shape'], inputs['scale_factor'])
                bbox_pred, bbox_num = bbox, bbox_num
            
            output = {
                "bbox": bbox_pred,
                "bbox_num": bbox_num,
            }
            return output