# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from modeling.bbox_utils import bbox_cxcywh_to_xyxy


class DETRBBoxPostProcess(object):
    def __init__(self,
                 num_classes=80,
                 num_top_queries=100,
                 use_focal_loss=False):
        super(DETRBBoxPostProcess, self).__init__()
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.use_focal_loss = use_focal_loss

    def __call__(self, head_out, im_shape, scale_factor):
        """
        Decode the bbox.

        Args:
            head_out (tuple): bbox_pred, cls_logit and masks of bbox_head output.
            im_shape (Tensor): The shape of the input image.
            scale_factor (Tensor): The scale factor of the input image.
        Returns:
            bbox_pred (Tensor): The output prediction with shape [N, 6], including
                labels, scores and bboxes. The size of bboxes are corresponding
                to the input image, the bboxes may be used in other branch.
            bbox_num (Tensor): The number of prediction boxes of each batch with
                shape [bs], and is N.
        """
        bboxes, logits, masks = head_out

        bbox_pred = bbox_cxcywh_to_xyxy(bboxes)
        origin_shape = paddle.floor(im_shape / scale_factor + 0.5)
        img_h, img_w = origin_shape.unbind(1)
        origin_shape = paddle.stack(
            [img_w, img_h, img_w, img_h], axis=-1).unsqueeze(0)
        bbox_pred *= origin_shape

        scores = F.sigmoid(logits) if self.use_focal_loss else F.softmax(
            logits)[:, :, :-1]

        if not self.use_focal_loss:
            scores, labels = scores.max(-1), scores.argmax(-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = paddle.topk(
                    scores, self.num_top_queries, axis=-1)
                labels = paddle.stack(
                    [paddle.gather(l, i) for l, i in zip(labels, index)])
                bbox_pred = paddle.stack(
                    [paddle.gather(b, i) for b, i in zip(bbox_pred, index)])
        else:
            scores, index = paddle.topk(
                scores.reshape([logits.shape[0], -1]),
                self.num_top_queries,
                axis=-1)
            labels = index % logits.shape[2]
            index = index // logits.shape[2]
            bbox_pred = paddle.stack(
                [paddle.gather(b, i) for b, i in zip(bbox_pred, index)])

        bbox_pred = paddle.concat(
            [
                labels.unsqueeze(-1).astype('float32'), scores.unsqueeze(-1),
                bbox_pred
            ],
            axis=-1)
        bbox_num = paddle.to_tensor(
            bbox_pred.shape[1], dtype='int32').tile([bbox_pred.shape[0]])
        bbox_pred = bbox_pred.reshape([-1, 6])
        return bbox_pred, bbox_num


def nms(dets, thresh):
    """Apply classic DPM-style greedy NMS."""
    if dets.shape[0] == 0:
        return dets[[], :]
    scores = dets[:, 0]
    x1 = dets[:, 1]
    y1 = dets[:, 2]
    x2 = dets[:, 3]
    y2 = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)

    # nominal indices
    # _i, _j
    # sorted indices
    # i, j
    # temp variables for box i's (the box currently under consideration)
    # ix1, iy1, ix2, iy2, iarea

    # variables for computing overlap with box j (lower scoring box)
    # xx1, yy1, xx2, yy2
    # w, h
    # inter, ovr

    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1
    keep = np.where(suppressed == 0)[0]
    dets = dets[keep, :]
    return dets