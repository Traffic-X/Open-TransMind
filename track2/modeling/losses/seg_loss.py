# !/usr/bin/env python3
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle import nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


class SegLoss(nn.Layer):
    def __init__(self):
        super(SegLoss, self).__init__()
        self.logits_list = []
        self.loss1 = CrossEntropyLoss()
        self.loss2 = RMILoss()
        self.loss21 = CrossEntropyLoss()
        self.loss3 = CrossEntropyLoss()
        self.loss4 = CrossEntropyLoss()

        self.losses = [self.loss1, self.loss2, self.loss21, self.loss3, self.loss4]
        self.coef = [0.4, 1.0, 1.0, 0.05, 0.05]
    
    def forward(self, output, labels):
        outputs = [output[0], output[1], output[1], output[2], output[3]]
        loss_list = self.loss_computation(outputs, labels, losses=self.losses, coef=self.coef)
        sum_loss = sum(loss_list)
        loss = dict()
        loss['crossentropy_loss'] = sum_loss[0] 
        return  loss


    def check_logits_losses(self, logits_list, losses):
        len_logits = len(logits_list)
        len_losses = len(losses)
        if len_logits != len_losses:
            raise RuntimeError(
                'The length of logits_list should equal to the types of loss config: {} != {}.'
                .format(len_logits, len_losses))


    def loss_computation(self, logits_list, labels, losses, coef):
        self.check_logits_losses(logits_list, losses)
        loss_list = []
        for i in range(len(logits_list)):
            logits = logits_list[i]
            loss_i = losses[i]
            coef_i = coef[i]
            loss_list.append(coef_i * loss_i(logits, labels))
        return loss_list


class SegBDD100KLoss(nn.Layer):
    def __init__(self, thresh=0.5, min_kept=10000, ignore_index=255):
        super(SegBDD100KLoss, self).__init__()
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index

        self.logits_list = []
        self.loss1 = OhemCrossEntropyLoss(thresh=self.thresh, min_kept=self.min_kept, ignore_index=self.ignore_index)
        self.loss2 = OhemCrossEntropyLoss(thresh=self.thresh, min_kept=self.min_kept, ignore_index=self.ignore_index)
        self.loss3 = OhemCrossEntropyLoss(thresh=self.thresh, min_kept=self.min_kept, ignore_index=self.ignore_index)

        self.losses = [self.loss1, self.loss2, self.loss3, ]
        self.coef = [1, 1, 1]
    
    def forward(self, outputs, labels):
        loss_list = self.loss_computation(outputs, labels, losses=self.losses, coef=self.coef)
        sum_loss = sum(loss_list)
        loss = dict()
        loss['OhemCrossEntropyLoss'] = sum_loss[0] 
        return  loss


    def check_logits_losses(self, logits_list, losses):
        len_logits = len(logits_list)
        len_losses = len(losses)
        if len_logits != len_losses:
            raise RuntimeError(
                'The length of logits_list should equal to the types of loss config: {} != {}.'
                .format(len_logits, len_losses))


    def loss_computation(self, logits_list, labels, losses, coef):
        self.check_logits_losses(logits_list, losses)
        loss_list = []
        for i in range(len(logits_list)):
            logits = logits_list[i]
            loss_i = losses[i]
            coef_i = coef[i]
            loss_list.append(coef_i * loss_i(logits, labels))
        return loss_list



class SegDMNetLoss(nn.Layer):
    def __init__(self, ):
        super(SegDMNetLoss, self).__init__()
  
        self.loss1 = CrossEntropyLoss()
        self.loss2 = CrossEntropyLoss()
        # self.loss3 = CrossEntropyLoss()

        self.losses = [self.loss1, self.loss2]
        self.coef = [1, 0.4]
    
    def forward(self, outputs, labels):
        loss_list = self.loss_computation(outputs, labels, losses=self.losses, coef=self.coef)
        sum_loss = sum(loss_list)
        # loss = dict()
        # loss['DMNetLoss1'] = loss_list[0] 
        # loss['DMNetLoss2'] = loss_list[1]
        # loss['DMNetLoss3'] = loss_list[2]
        return  sum_loss


    def check_logits_losses(self, logits_list, losses):
        len_logits = len(logits_list)
        len_losses = len(losses)
        if len_logits != len_losses:
            raise RuntimeError(
                'The length of logits_list should equal to the types of loss config: {} != {}.'
                .format(len_logits, len_losses))


    def loss_computation(self, logits_list, labels, losses, coef):
        self.check_logits_losses(logits_list, losses)
        loss_list = []
        for i in range(len(logits_list)):
            logits = logits_list[i]
            loss_i = losses[i]
            coef_i = coef[i]
            loss_list.append(coef_i * loss_i(logits, labels))
        return loss_list



class SegSETRLoss(nn.Layer):
    def __init__(self, ):
        super(SegSETRLoss, self).__init__()
  
        self.loss1 = CrossEntropyLoss()
        self.loss2 = CrossEntropyLoss()
        self.loss3 = CrossEntropyLoss()
        self.loss4 = CrossEntropyLoss()
        self.loss5 = CrossEntropyLoss()
        self.losses = [self.loss1, self.loss2, self.loss5, self.loss5, self.loss5]
        self.coef = [1, 0.4, 0.4, 0.4, 0.4]
    
    def forward(self, outputs, labels):
        loss_list = self.loss_computation(outputs, labels, losses=self.losses, coef=self.coef)
        loss = dict()
        # loss['SegSETRLoss1'] = loss_list[0] 
        # loss['SegSETRLoss2'] = loss_list[1]
        # loss['SegSETRLoss3'] = loss_list[2]
        # loss['SegSETRLoss4'] = loss_list[3]
        # loss['SegSETRLoss5'] = loss_list[4]
        loss['seg_setr_loss'] = sum(loss_list)
        return  loss


    def check_logits_losses(self, logits_list, losses):
        len_logits = len(logits_list)
        len_losses = len(losses)
        if len_logits != len_losses:
            raise RuntimeError(
                'The length of logits_list should equal to the types of loss config: {} != {}.'
                .format(len_logits, len_losses))


    def loss_computation(self, logits_list, labels, losses, coef):
        self.check_logits_losses(logits_list, losses)
        loss_list = []
        for i in range(len(logits_list)):
            logits = logits_list[i]
            loss_i = losses[i]
            coef_i = coef[i]
            loss_list.append(coef_i * loss_i(logits, labels))
        return loss_list


class CrossEntropyLoss(nn.Layer):
    """
    Implements the cross entropy loss function.

    Args:
        weight (tuple|list|ndarray|Tensor, optional): A manual rescaling weight
            given to each class. Its length must be equal to the number of classes.
            Default ``None``.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        top_k_percent_pixels (float, optional): the value lies in [0.0, 1.0].
            When its value < 1.0, only compute the loss for the top k percent pixels
            (e.g., the top 20% pixels). This is useful for hard pixel mining. Default ``1.0``.
        data_format (str, optional): The tensor format to use, 'NCHW' or 'NHWC'. Default ``'NCHW'``.
    """

    def __init__(self,
                 weight=None,
                 ignore_index=255,
                 top_k_percent_pixels=1.0,
                 data_format='NCHW'):
        super(CrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.top_k_percent_pixels = top_k_percent_pixels
        self.EPS = 1e-8
        self.data_format = data_format
        if weight is not None:
            self.weight = paddle.to_tensor(weight, dtype='float32')
        else:
            self.weight = None

    def forward(self, logit, label, semantic_weights=None):
        """
        Forward computation.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
            semantic_weights (Tensor, optional): Weights about loss for each pixels,
                shape is the same as label. Default: None.
        Returns:
            (Tensor): The average loss.
        """
        channel_axis = 1 if self.data_format == 'NCHW' else -1
        if self.weight is not None and logit.shape[channel_axis] != len(
                self.weight):
            raise ValueError(
                'The number of weights = {} must be the same as the number of classes = {}.'
                .format(len(self.weight), logit.shape[channel_axis]))

        if channel_axis == 1:
            logit = paddle.transpose(logit, [0, 2, 3, 1])
        label = label.astype('int64')

        loss = F.cross_entropy(
            logit,
            label,
            ignore_index=self.ignore_index,
            reduction='none',
            weight=self.weight)

        return self._post_process_loss(logit, label, semantic_weights, loss)

    def _post_process_loss(self, logit, label, semantic_weights, loss):
        """
        Consider mask and top_k to calculate the final loss.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
            semantic_weights (Tensor, optional): Weights about loss for each pixels,
                shape is the same as label.
            loss (Tensor): Loss tensor which is the output of cross_entropy. If soft_label
                is False in cross_entropy, the shape of loss should be the same as the label.
                If soft_label is True in cross_entropy, the shape of loss should be
                (N, D1, D2,..., Dk, 1).
        Returns:
            (Tensor): The average loss.
        """
        mask = label != self.ignore_index
        mask = paddle.cast(mask, 'float32')
        label.stop_gradient = True
        mask.stop_gradient = True

        if loss.ndim > mask.ndim:
            loss = paddle.squeeze(loss, axis=-1)
        loss = loss * mask
        if semantic_weights is not None:
            loss = loss * semantic_weights

        if self.weight is not None:
            _one_hot = F.one_hot(label * mask, logit.shape[-1])
            coef = paddle.sum(_one_hot * self.weight, axis=-1)
        else:
            coef = paddle.ones_like(label)

        if self.top_k_percent_pixels == 1.0:
            avg_loss = paddle.mean(loss) / (paddle.mean(mask * coef) + self.EPS)
        else:
            loss = loss.reshape((-1, ))
            top_k_pixels = int(self.top_k_percent_pixels * loss.numel())
            loss, indices = paddle.topk(loss, top_k_pixels)
            coef = coef.reshape((-1, ))
            coef = paddle.gather(coef, indices)
            coef.stop_gradient = True
            coef = coef.astype('float32')
            avg_loss = loss.mean() / (paddle.mean(coef) + self.EPS)

        return avg_loss


class DistillCrossEntropyLoss(CrossEntropyLoss):
    """
    The implementation of distill cross entropy loss.

    Args:
        weight (tuple|list|ndarray|Tensor, optional): A manual rescaling weight
            given to each class. Its length must be equal to the number of classes.
            Default ``None``.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
        top_k_percent_pixels (float, optional): the value lies in [0.0, 1.0].
            When its value < 1.0, only compute the loss for the top k percent pixels
            (e.g., the top 20% pixels). This is useful for hard pixel mining.
            Default ``1.0``.
        data_format (str, optional): The tensor format to use, 'NCHW' or 'NHWC'.
            Default ``'NCHW'``.
    """

    def __init__(self,
                 weight=None,
                 ignore_index=255,
                 top_k_percent_pixels=1.0,
                 data_format='NCHW'):
        super().__init__(weight, ignore_index, top_k_percent_pixels,
                         data_format)

    def forward(self,
                student_logit,
                teacher_logit,
                label,
                semantic_weights=None):
        """
        Forward computation.

        Args:
            student_logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            teacher_logit (Tensor): Logit tensor, the data type is float32, float64. The shape
                is the same as the student_logit.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
            semantic_weights (Tensor, optional): Weights about loss for each pixels,
                shape is the same as label. Default: None.
        """

        if student_logit.shape != teacher_logit.shape:
            raise ValueError(
                'The shape of student_logit = {} must be the same as the shape of teacher_logit = {}.'
                .format(student_logit.shape, teacher_logit.shape))

        channel_axis = 1 if self.data_format == 'NCHW' else -1
        if self.weight is not None and student_logit.shape[channel_axis] != len(
                self.weight):
            raise ValueError(
                'The number of weights = {} must be the same as the number of classes = {}.'
                .format(len(self.weight), student_logit.shape[channel_axis]))

        if channel_axis == 1:
            student_logit = paddle.transpose(student_logit, [0, 2, 3, 1])
            teacher_logit = paddle.transpose(teacher_logit, [0, 2, 3, 1])

        teacher_logit = F.softmax(teacher_logit)

        loss = F.cross_entropy(
            student_logit,
            teacher_logit,
            weight=self.weight,
            reduction='none',
            soft_label=True)

        return self._post_process_loss(student_logit, label, semantic_weights,
                                       loss)

class MixedLoss(nn.Layer):
    """
    Weighted computations for multiple Loss.
    The advantage is that mixed loss training can be achieved without changing the networking code.

    Args:
        losses (list[nn.Layer]): A list consisting of multiple loss classes
        coef (list[float|int]): Weighting coefficient of multiple loss

    Returns:
        A callable object of MixedLoss.
    """

    def __init__(self, losses, coef):
        super(MixedLoss, self).__init__()
        if not isinstance(losses, list):
            raise TypeError('`losses` must be a list!')
        if not isinstance(coef, list):
            raise TypeError('`coef` must be a list!')
        len_losses = len(losses)
        len_coef = len(coef)
        if len_losses != len_coef:
            raise ValueError(
                'The length of `losses` should equal to `coef`, but they are {} and {}.'
                .format(len_losses, len_coef))

        self.losses = losses
        self.coef = coef

    def forward(self, logits, labels):
        loss_list = []
        for i, loss in enumerate(self.losses):
            output = loss(logits, labels)
            loss_list.append(output * self.coef[i])
        return loss_list




_euler_num = 2.718281828
_pi = 3.14159265
_ln_2_pi = 1.837877
_CLIP_MIN = 1e-6
_CLIP_MAX = 1.0
_POS_ALPHA = 5e-4
_IS_SUM = 1


class RMILoss(nn.Layer):
    """
    Implements the Region Mutual Information(RMI) Loss（https://arxiv.org/abs/1910.12037） for Semantic Segmentation.
    Unlike vanilla rmi loss which contains Cross Entropy Loss, we disband them and only
    left the RMI-related parts.
    The motivation is to allow for a more flexible combination of losses during training.
    For example, by employing mixed loss to merge RMI Loss with Boostrap Cross Entropy Loss,
    we can achieve the online mining of hard examples together with attention to region information.
    Args:
        weight (tuple|list|ndarray|Tensor, optional): A manual rescaling weight
            given to each class. Its length must be equal to the number of classes.
            Default ``None``.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self,
                 num_classes=19,
                 rmi_radius=3,
                 rmi_pool_way=0,
                 rmi_pool_size=3,
                 rmi_pool_stride=3,
                 loss_weight_lambda=0.5,
                 ignore_index=255):
        super(RMILoss, self).__init__()

        self.num_classes = num_classes
        assert rmi_radius in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.rmi_radius = rmi_radius
        assert rmi_pool_way in [0, 1, 2, 3]
        self.rmi_pool_way = rmi_pool_way
        assert rmi_pool_size == rmi_pool_stride
        self.rmi_pool_size = rmi_pool_size
        self.rmi_pool_stride = rmi_pool_stride
        self.weight_lambda = loss_weight_lambda
        self.half_d = self.rmi_radius * self.rmi_radius
        self.d = 2 * self.half_d
        self.kernel_padding = self.rmi_pool_size // 2
        self.ignore_index = ignore_index

    def forward(self, logits_4D, labels_4D, do_rmi=True):
        """
        Forward computation.
        Args:
            logits (Tensor): Shape is [N, C, H, W], logits at each prediction (between -\infty and +\infty).
            labels (Tensor): Shape is [N, H, W], ground truth labels (between 0 and C - 1).
        """
        logits_4D = paddle.cast(logits_4D, dtype='float32')
        labels_4D = paddle.cast(labels_4D, dtype='float32')

        loss = self.forward_sigmoid(logits_4D, labels_4D, do_rmi=do_rmi)
        return loss

    def forward_sigmoid(self, logits_4D, labels_4D, do_rmi=False):
        """
        Using the sigmiod operation both.
        Args:
                logits_4D   :   [N, C, H, W], dtype=float32
                labels_4D   :   [N, H, W], dtype=long
                do_rmi          :       bool
        """
        label_mask_3D = labels_4D != self.ignore_index
        valid_onehot_labels_4D = paddle.cast(
            F.one_hot(
                paddle.cast(
                    labels_4D, dtype='int64') * paddle.cast(
                        label_mask_3D, dtype='int64'),
                num_classes=self.num_classes),
            dtype='float32')
        # label_mask_flat = paddle.cast(
        #     paddle.reshape(label_mask_3D, [-1]), dtype='float32')

        valid_onehot_labels_4D = valid_onehot_labels_4D * paddle.unsqueeze(
            label_mask_3D, axis=3)
        valid_onehot_labels_4D.stop_gradient = True
        probs_4D = F.sigmoid(logits_4D) * paddle.unsqueeze(
            label_mask_3D, axis=1) + _CLIP_MIN

        valid_onehot_labels_4D = paddle.transpose(valid_onehot_labels_4D,
                                                  [0, 3, 1, 2])
        valid_onehot_labels_4D.stop_gradient = True
        rmi_loss = self.rmi_lower_bound(valid_onehot_labels_4D, probs_4D)

        return rmi_loss

    def inverse(self, x):
        return paddle.inverse(x)

    def rmi_lower_bound(self, labels_4D, probs_4D):
        """
        calculate the lower bound of the region mutual information.
        Args:
                labels_4D   :   [N, C, H, W], dtype=float32
                probs_4D    :   [N, C, H, W], dtype=float32
        """
        assert labels_4D.shape == probs_4D.shape, print(
            'shapes', labels_4D.shape, probs_4D.shape)

        p, s = self.rmi_pool_size, self.rmi_pool_stride
        if self.rmi_pool_stride > 1:
            if self.rmi_pool_way == 0:
                labels_4D = F.max_pool2d(
                    labels_4D,
                    kernel_size=p,
                    stride=s,
                    padding=self.kernel_padding)
                probs_4D = F.max_pool2d(
                    probs_4D,
                    kernel_size=p,
                    stride=s,
                    padding=self.kernel_padding)
            elif self.rmi_pool_way == 1:
                labels_4D = F.avg_pool2d(
                    labels_4D,
                    kernel_size=p,
                    stride=s,
                    padding=self.kernel_padding)
                probs_4D = F.avg_pool2d(
                    probs_4D,
                    kernel_size=p,
                    stride=s,
                    padding=self.kernel_padding)
            elif self.rmi_pool_way == 2:
                shape = labels_4D.shape
                new_h, new_w = shape[2] // s, shape[3] // s
                labels_4D = F.interpolate(
                    labels_4D, size=[new_h, new_w], mode='nearest')
                probs_4D = F.interpolate(
                    probs_4D,
                    size=[new_h, new_w],
                    mode='bilinear',
                    align_corners=True)
            else:
                raise NotImplementedError("Pool way of RMI is not defined!")

        label_shape = labels_4D.shape
        n, c = label_shape[0], label_shape[1]

        la_vectors, pr_vectors = self.map_get_pairs(
            labels_4D, probs_4D, radius=self.rmi_radius, is_combine=0)

        la_vectors = paddle.reshape(la_vectors, [n, c, self.half_d, -1])
        la_vectors = paddle.cast(la_vectors, dtype='float64')
        la_vectors.stop_gradient = True

        pr_vectors = paddle.reshape(pr_vectors, [n, c, self.half_d, -1])
        pr_vectors = paddle.cast(pr_vectors, dtype='float64')

        diag_matrix = paddle.unsqueeze(
            paddle.unsqueeze(
                paddle.eye(self.half_d), axis=0), axis=0)
        la_vectors = la_vectors - paddle.mean(la_vectors, axis=3, keepdim=True)

        la_cov = paddle.matmul(la_vectors,
                               paddle.transpose(la_vectors, [0, 1, 3, 2]))
        pr_vectors = pr_vectors - paddle.mean(pr_vectors, axis=3, keepdim=True)
        pr_cov = paddle.matmul(pr_vectors,
                               paddle.transpose(pr_vectors, [0, 1, 3, 2]))

        pr_cov_inv = self.inverse(pr_cov + paddle.cast(
            diag_matrix, dtype='float64') * _POS_ALPHA)

        la_pr_cov = paddle.matmul(la_vectors,
                                  paddle.transpose(pr_vectors, [0, 1, 3, 2]))

        appro_var = la_cov - paddle.matmul(
            paddle.matmul(la_pr_cov, pr_cov_inv),
            paddle.transpose(la_pr_cov, [0, 1, 3, 2]))

        rmi_now = 0.5 * self.log_det_by_cholesky(appro_var + paddle.cast(
            diag_matrix, dtype='float64') * _POS_ALPHA)

        rmi_per_class = paddle.cast(
            paddle.mean(
                paddle.reshape(rmi_now, [-1, self.num_classes]), axis=0),
            dtype='float32')
        rmi_per_class = paddle.divide(rmi_per_class,
                                      paddle.to_tensor(float(self.half_d)))

        rmi_loss = paddle.sum(rmi_per_class) if _IS_SUM else paddle.mean(
            rmi_per_class)

        return rmi_loss

    def log_det_by_cholesky(self, matrix):
        """
        Args:
            matrix: matrix must be a positive define matrix.
                    shape [N, C, D, D].
        """

        chol = paddle.cholesky(matrix)
        diag = paddle.diagonal(chol, offset=0, axis1=-2, axis2=-1)
        chol = paddle.log(diag + 1e-8)

        return 2.0 * paddle.sum(chol, axis=-1)

    def map_get_pairs(self, labels_4D, probs_4D, radius=3, is_combine=True):
        """
        Args:
            labels_4D   :   labels, shape [N, C, H, W]
            probs_4D    :   probabilities, shape [N, C, H, W]
            radius      :   the square radius
        Return:
            tensor with shape [N, C, radius * radius, H - (radius - 1), W - (radius - 1)]
        """

        label_shape = labels_4D.shape
        h, w = label_shape[2], label_shape[3]
        new_h, new_w = h - (radius - 1), w - (radius - 1)
        la_ns = []
        pr_ns = []
        for y in range(0, radius, 1):
            for x in range(0, radius, 1):
                la_now = labels_4D[:, :, y:y + new_h, x:x + new_w]
                pr_now = probs_4D[:, :, y:y + new_h, x:x + new_w]
                la_ns.append(la_now)
                pr_ns.append(pr_now)

        if is_combine:
            pair_ns = la_ns + pr_ns
            p_vectors = paddle.stack(pair_ns, axis=2)
            return p_vectors
        else:
            la_vectors = paddle.stack(la_ns, axis=2)
            pr_vectors = paddle.stack(pr_ns, axis=2)
            return la_vectors, pr_vectors



class OhemCrossEntropyLoss(nn.Layer):
    """
    Implements the ohem cross entropy loss function.

    Args:
        thresh (float, optional): The threshold of ohem. Default: 0.7.
        min_kept (int, optional): The min number to keep in loss computation. Default: 10000.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, thresh=0.5, min_kept=10000, ignore_index=255):
        super(OhemCrossEntropyLoss, self).__init__()
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        self.EPS = 1e-5

    def forward(self, logit, label):
        """
        Forward computation.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
        """
        if len(label.shape) != len(logit.shape):
            label = paddle.unsqueeze(label, 1)

        # get the label after ohem
        n, c, h, w = logit.shape
        label = label.reshape((-1, )).astype('int64')
        valid_mask = (label != self.ignore_index).astype('int64')
        num_valid = valid_mask.sum()
        label = label * valid_mask

        prob = F.softmax(logit, axis=1)
        prob = prob.transpose((1, 0, 2, 3)).reshape((c, -1))

        if self.min_kept < num_valid and num_valid > 0:
            # let the value which ignored greater than 1
            prob = prob + (1 - valid_mask)

            # get the prob of relevant label
            label_onehot = F.one_hot(label, c)
            label_onehot = label_onehot.transpose((1, 0))
            prob = prob * label_onehot
            prob = paddle.sum(prob, axis=0)

            threshold = self.thresh
            if self.min_kept > 0:
                index = prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                threshold_index = int(threshold_index)
                if prob[threshold_index] > self.thresh:
                    threshold = prob[threshold_index]
                kept_mask = (prob < threshold).astype('int64')
                label = label * kept_mask
                valid_mask = valid_mask * kept_mask

        # make the invalid region as ignore
        label = label + (1 - valid_mask) * self.ignore_index

        label = label.reshape((n, 1, h, w))
        valid_mask = valid_mask.reshape((n, 1, h, w)).astype('float32')
        loss = F.softmax_with_cross_entropy(
            logit, label, ignore_index=self.ignore_index, axis=1)
        loss = loss * valid_mask
        avg_loss = paddle.mean(loss) / (paddle.mean(valid_mask) + self.EPS)

        label.stop_gradient = True
        valid_mask.stop_gradient = True
        return avg_loss