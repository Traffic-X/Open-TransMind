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
import pycocotools.mask as mask_util
from modeling.initializer import linear_init_, constant_


class FasterRCNNHead(nn.Layer):
    """
    """
    def __init__(self, neck, rpn_head, bbox_head, bbox_post_process):
        super().__init__()
        self.neck=neck
        self.rpn_head = rpn_head
        self.bbox_head = bbox_head
        self.bbox_post_process = bbox_post_process
    
    def forward(self, body_feats, inputs):
        body_feats = self.neck(body_feats)
        if 'gt_bbox' in inputs:
            #hard code
            gt_bbox = [paddle.cast(inputs['gt_bbox'][i], 'float32') for i in range(len(inputs['gt_bbox']))]
            inputs['gt_bbox'] = gt_bbox
        if self.training:
            rois, rois_num, rpn_loss = self.rpn_head(body_feats, inputs)
            bbox_loss, _ = self.bbox_head(body_feats, rois, rois_num,
                                          inputs)
            loss = {}
            loss.update(rpn_loss)
            loss.update(bbox_loss)
            total_loss = paddle.add_n(list(loss.values()))
            loss.update({'loss': total_loss})
            return loss

        else:
            rois, rois_num, _ = self.rpn_head(body_feats, inputs)
            preds, _ = self.bbox_head(body_feats, rois, rois_num, None)

            im_shape = inputs['im_shape']
            scale_factor = inputs['scale_factor']
            bbox, bbox_num = self.bbox_post_process(preds, (rois, rois_num),
                                                    im_shape, scale_factor)

            # rescale the prediction back to origin image
            bboxes, bbox_pred, bbox_num = self.bbox_post_process.get_pred(
                bbox, bbox_num, im_shape, scale_factor)
            output = {'bbox': bbox_pred, 'bbox_num': bbox_num}
            return output