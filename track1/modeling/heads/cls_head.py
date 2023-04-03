from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import paddle.nn as nn
from paddle.nn import AdaptiveAvgPool2D
from modeling.losses import cross_entropy_loss
import paddle
from paddleseg.models import layers

from .cbam import CBAM_Module
 

class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 is_vd_mode=False,
                 act=None,
                 data_format='NCHW'):
        super(ConvBNLayer, self).__init__()
        if dilation != 1 and kernel_size != 3:
            raise RuntimeError("When the dilation isn't 1," \
                "the kernel_size should be 3.")

        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = nn.AvgPool2D(
            kernel_size=2,
            stride=2,
            padding=0,
            ceil_mode=True,
            data_format=data_format)
        self._conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 \
                if dilation == 1 else dilation,
            dilation=dilation,
            groups=groups,
            bias_attr=False,
            data_format=data_format)

        self._batch_norm = layers.SyncBatchNorm(
            out_channels, data_format=data_format)
        self._act_op = layers.Activation(act=act)

    def forward(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        y = self._act_op(y)

        return y


class ClsHead(nn.Layer):
    def __init__(self, embedding_size, class_num, **kwargs):
        super(ClsHead, self).__init__()
        self.embedding_size = embedding_size
        self.class_num = class_num
 
        self.avg_pool = AdaptiveAvgPool2D(1, data_format="NCHW")

        self.conv16_32 = ConvBNLayer(self.embedding_size, self.embedding_size, 3, 2, 1, act="relu")

        self.conv1 = ConvBNLayer(self.embedding_size * 2, self.embedding_size * 2, 3, 1, 1, act="relu")
        self.conv2 =  ConvBNLayer(self.embedding_size * 2, self.embedding_size * 2, 3, 1, 1, act="relu")
 
        self.fc1 = nn.Linear(self.embedding_size * 2, self.embedding_size)
        self.fc2 = nn.Linear(self.embedding_size, self.class_num)
        self.flatten = nn.Flatten()

        self.cbam = CBAM_Module(self.embedding_size * 2)

    def forward(self, inputs, targets=None):
        if isinstance(inputs, list):
            inputs = inputs[0][2:4]     # 0-transformer 1-cnn   
        else:
            inputs = inputs[1]
        gt_labels = targets['targets']

        feature16 = inputs[0]
        feature32 = inputs[1]
        feature16_32 = self.conv16_32(feature16)
        features = paddle.concat([feature16_32, feature32], axis=1)

        input = self.conv1(features)
        input = self.conv2(input)

        input = self.cbam(input)

        inputs = self.avg_pool(input)
        inputs = self.flatten(inputs)
        outputs = self.fc2(self.fc1(inputs))
        if self.training:
            return self.get_loss(outputs, gt_labels)
        else:
            return outputs
 
    def get_loss(self, outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        loss_dict = {}
        loss_dict['loss_ce_cls'] = cross_entropy_loss(
                outputs,
                gt_labels,
                0.1
            )
 
        return loss_dict