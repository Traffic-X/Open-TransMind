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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from scipy.optimize import linear_sum_assignment

from modeling.losses.iou_loss import GIoULoss
from modeling.bbox_utils import bbox_cxcywh_to_xyxy


__all__ = ['HungarianMatcher', 'DETRLoss']


def sigmoid_focal_loss(logit, label, normalizer=1.0, alpha=0.25, gamma=2.0):
    prob = F.sigmoid(logit)
    ce_loss = F.binary_cross_entropy_with_logits(logit, label, reduction="none")
    p_t = prob * label + (1 - prob) * (1 - label)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * label + (1 - alpha) * (1 - label)
        loss = alpha_t * loss
    return loss.mean(1).sum() / normalizer


class HungarianMatcher(nn.Layer):
    def __init__(self,
                 matcher_coeff={'class': 1,
                                'bbox': 5,
                                'giou': 2},
                 use_focal_loss=False,
                 alpha=0.25,
                 gamma=2.0):
        r"""
        Args:
            matcher_coeff (dict): The coefficient of hungarian matcher cost.
        """
        super(HungarianMatcher, self).__init__()
        self.matcher_coeff = matcher_coeff
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        self.giou_loss = GIoULoss()

    def forward(self, boxes, logits, gt_bbox, gt_class):
        r"""
        Args:
            boxes (Tensor): [b, query, 4]
            logits (Tensor): [b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = boxes.shape[:2]

        num_gts = sum(len(a) for a in gt_class)
        if num_gts == 0:
            return [(paddle.to_tensor(
                [], dtype=paddle.int64), paddle.to_tensor(
                    [], dtype=paddle.int64)) for _ in range(bs)]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]
        out_prob = F.sigmoid(logits.flatten(
            0, 1)) if self.use_focal_loss else F.softmax(logits.flatten(0, 1))
        # [batch_size * num_queries, 4]
        out_bbox = boxes.flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = paddle.concat(gt_class).flatten()
        tgt_bbox = paddle.concat(gt_bbox)

        # Compute the classification cost
        if self.use_focal_loss:
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(
                1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * (
                (1 - out_prob)**self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = paddle.gather(
                pos_cost_class, tgt_ids, axis=1) - paddle.gather(
                    neg_cost_class, tgt_ids, axis=1)
        else:
            cost_class = -paddle.gather(out_prob, tgt_ids, axis=1)

        # Compute the L1 cost between boxes
        cost_bbox = (
            out_bbox.unsqueeze(1) - tgt_bbox.unsqueeze(0)).abs().sum(-1)

        # Compute the giou cost betwen boxes
        cost_giou = self.giou_loss(
            bbox_cxcywh_to_xyxy(out_bbox.unsqueeze(1)),
            bbox_cxcywh_to_xyxy(tgt_bbox.unsqueeze(0))).squeeze(-1)

        # Final cost matrix
        C = self.matcher_coeff['class'] * cost_class + self.matcher_coeff['bbox'] * cost_bbox + \
            self.matcher_coeff['giou'] * cost_giou
        C = C.reshape([bs, num_queries, -1])
        C = [a.squeeze(0) for a in C.chunk(bs)]

        sizes = [a.shape[0] for a in gt_bbox]
        indices = [
            linear_sum_assignment(c.split(sizes, -1)[i].numpy())
            for i, c in enumerate(C)
        ]
        return [(paddle.to_tensor(
            i, dtype=paddle.int64), paddle.to_tensor(
                j, dtype=paddle.int64)) for i, j in indices]


class DETRLoss(nn.Layer):
    def __init__(self,
                 num_classes=80,
                 matcher_coeff={'class': 1,
                                'bbox': 5,
                                'giou': 2},
                 loss_coeff={
                     'class': 1,
                     'bbox': 5,
                     'giou': 2,
                     'no_object': 0.1,
                     'mask': 1,
                     'dice': 1
                 },
                 aux_loss=True,
                 use_focal_loss=False):
        r"""
        Args:
            num_classes (int): The number of classes.
            matcher (HungarianMatcher): It computes an assignment between the targets
                and the predictions of the network.
            loss_coeff (dict): The coefficient of loss.
            aux_loss (bool): If 'aux_loss = True', loss at each decoder layer are to be used.
            use_focal_loss (bool): Use focal loss or not.
        """
        super(DETRLoss, self).__init__()
        self.num_classes = num_classes

        self.matcher = HungarianMatcher(matcher_coeff=matcher_coeff)
        self.loss_coeff = loss_coeff
        self.aux_loss = aux_loss
        self.use_focal_loss = use_focal_loss

        if not self.use_focal_loss:
            self.loss_coeff['class'] = paddle.full([num_classes + 1],
                                                   loss_coeff['class'])
            self.loss_coeff['class'][-1] = loss_coeff['no_object']
        
        self.giou_loss = GIoULoss()

    def _get_loss_class(self, logits, gt_class, match_indices, bg_index,
                        num_gts):
        # logits: [b, query, num_classes], gt_class: list[[n, 1]]
        target_label = paddle.full(logits.shape[:2], bg_index, dtype='int64')
        bs, num_query_objects = target_label.shape
        if sum(len(a) for a in gt_class) > 0:
            index, updates = self._get_index_updates(num_query_objects,
                                                     gt_class, match_indices)
            target_label = paddle.scatter(
                target_label.reshape([-1, 1]), index, updates.astype('int64'))
            target_label = target_label.reshape([bs, num_query_objects])
        if self.use_focal_loss:
            target_label = F.one_hot(target_label,
                                     self.num_classes + 1)[..., :-1]
        return {
            'loss_class': self.loss_coeff['class'] * sigmoid_focal_loss(
                logits, target_label, num_gts / num_query_objects)
            if self.use_focal_loss else F.cross_entropy(
                logits, target_label, weight=self.loss_coeff['class'])
        }

    def _get_loss_bbox(self, boxes, gt_bbox, match_indices, num_gts):
        # boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        loss = dict()
        if sum(len(a) for a in gt_bbox) == 0:
            loss['loss_bbox'] = paddle.to_tensor([0.])
            loss['loss_giou'] = paddle.to_tensor([0.])
            return loss

        src_bbox, target_bbox = self._get_src_target_assign(boxes, gt_bbox,
                                                            match_indices)
        loss['loss_bbox'] = self.loss_coeff['bbox'] * F.l1_loss(
            src_bbox, target_bbox, reduction='sum') / num_gts
        loss['loss_giou'] = self.giou_loss(
            bbox_cxcywh_to_xyxy(src_bbox), bbox_cxcywh_to_xyxy(target_bbox))
        loss['loss_giou'] = loss['loss_giou'].sum() / num_gts
        loss['loss_giou'] = self.loss_coeff['giou'] * loss['loss_giou']
        return loss

    def _get_loss_mask(self, masks, gt_mask, match_indices, num_gts):
        # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
        loss = dict()
        if sum(len(a) for a in gt_mask) == 0:
            loss['loss_mask'] = paddle.to_tensor([0.])
            loss['loss_dice'] = paddle.to_tensor([0.])
            return loss

        src_masks, target_masks = self._get_src_target_assign(masks, gt_mask,
                                                              match_indices)
        src_masks = F.interpolate(
            src_masks.unsqueeze(0),
            size=target_masks.shape[-2:],
            mode="bilinear")[0]
        loss['loss_mask'] = self.loss_coeff['mask'] * F.sigmoid_focal_loss(
            src_masks,
            target_masks,
            paddle.to_tensor(
                [num_gts], dtype='float32'))
        loss['loss_dice'] = self.loss_coeff['dice'] * self._dice_loss(
            src_masks, target_masks, num_gts)
        return loss

    def _dice_loss(self, inputs, targets, num_gts):
        inputs = F.sigmoid(inputs)
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_gts

    def _get_loss_aux(self, boxes, logits, gt_bbox, gt_class, bg_index,
                      num_gts):
        loss_class = []
        loss_bbox = []
        loss_giou = []
        for aux_boxes, aux_logits in zip(boxes, logits):
            match_indices = self.matcher(aux_boxes, aux_logits, gt_bbox,
                                         gt_class)
            loss_class.append(
                self._get_loss_class(aux_logits, gt_class, match_indices,
                                     bg_index, num_gts)['loss_class'])
            loss_ = self._get_loss_bbox(aux_boxes, gt_bbox, match_indices,
                                        num_gts)
            loss_bbox.append(loss_['loss_bbox'])
            loss_giou.append(loss_['loss_giou'])
        loss = {
            'loss_class_aux': paddle.add_n(loss_class),
            'loss_bbox_aux': paddle.add_n(loss_bbox),
            'loss_giou_aux': paddle.add_n(loss_giou)
        }
        return loss

    def _get_index_updates(self, num_query_objects, target, match_indices):
        batch_idx = paddle.concat([
            paddle.full_like(src, i) for i, (src, _) in enumerate(match_indices)
        ])
        src_idx = paddle.concat([src for (src, _) in match_indices])
        src_idx += (batch_idx * num_query_objects)
        target_assign = paddle.concat([
            paddle.gather(
                t, dst, axis=0) for t, (_, dst) in zip(target, match_indices)
        ])
        return src_idx, target_assign

    def _get_src_target_assign(self, src, target, match_indices):
        src_assign = paddle.concat([
            paddle.gather(
                t, I, axis=0) if len(I) > 0 else paddle.zeros([0, t.shape[-1]])
            for t, (I, _) in zip(src, match_indices)
        ])
        target_assign = paddle.concat([
            paddle.gather(
                t, J, axis=0) if len(J) > 0 else paddle.zeros([0, t.shape[-1]])
            for t, (_, J) in zip(target, match_indices)
        ])
        return src_assign, target_assign

    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                masks=None,
                gt_mask=None):
        r"""
        Args:
            boxes (Tensor): [l, b, query, 4]
            logits (Tensor): [l, b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]
            masks (Tensor, optional): [b, query, h, w]
            gt_mask (List(Tensor), optional): list[[n, H, W]]
        """
        match_indices = self.matcher(boxes[-1].detach(), logits[-1].detach(),
                                     gt_bbox, gt_class)
        num_gts = sum(len(a) for a in gt_bbox)
        try:
            # TODO: Paddle does not have a "paddle.distributed.is_initialized()"
            num_gts = paddle.to_tensor([num_gts], dtype=paddle.float32)
            paddle.distributed.all_reduce(num_gts)
            num_gts = paddle.clip(
                num_gts / paddle.distributed.get_world_size(), min=1).item()
        except:
            num_gts = max(num_gts.item(), 1)
        total_loss = dict()
        total_loss.update(
            self._get_loss_class(logits[-1], gt_class, match_indices,
                                 self.num_classes, num_gts))
        total_loss.update(
            self._get_loss_bbox(boxes[-1], gt_bbox, match_indices, num_gts))
        if masks is not None and gt_mask is not None:
            total_loss.update(
                self._get_loss_mask(masks, gt_mask, match_indices, num_gts))

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(boxes[:-1], logits[:-1], gt_bbox, gt_class,
                                   self.num_classes, num_gts))

        return total_loss


def build_loss_detr_lazy(matcher_coeff={'class': 1, 'bbox': 5, 'giou': 2},
    loss_coeff={'class': 1, 'bbox': 5, 'giou': 2, 'no_object': 0.1, 'mask': 1, 'dice': 1},
    aux_loss=True, **kwargs):
    """
    Create a detr loss instance from config
    """
    model = DETRLoss(matcher_coeff=matcher_coeff, loss_coeff=loss_coeff, aux_loss=aux_loss)
    return model