import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager
from paddleseg.models import layers
from paddleseg.utils import utils


class DMNetV2(nn.Layer):
    """
    The DMNet implementation based on PaddlePaddle.

    The original article refers to
     Junjun He, Zhongying Deng, Yu Qiao. "Dynamic Multi-scale Filters for Semantic Segmentation"

    Args:
        num_classes (int): The unique number of target classes.
        backbone (paddle.nn.Layer): Backbone network, currently support Resnet50_vd/Resnet101_vd.
        mid_channels (int): The middle channels of convolution layer. Default: 512.
        filter_sizes (list, tuple): The filter size of generated convolution kernel used in Dynamic Convolutional Module. Default: [1, 3, 5, 7].
        fusion (bool): Add one conv to fuse DCM output feature. Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone_dim=768,
                 mid_channels=512,
                 filter_sizes=[1, 3, 5, 7],
                 backbone_layeer=50,
                 num_filters = [64, 128, 256, 512],
                 fusion=False,
                 loss="SegDMNetLoss",):
        super().__init__()
        self.backbone_dim = backbone_dim
        self.loss = loss
        self.dcm_modules = nn.LayerList()
        self.feat_channels = [c * 4 for c in num_filters
                              ] if backbone_layeer >= 50 else num_filters
        for filter_size in filter_sizes:
            self.dcm_modules.append(
                DCM(filter_size, fusion, self.feat_channels[-1],
                    mid_channels), )
        self.bottleneck = layers.ConvBNReLU(
            self.feat_channels[-1] + len(filter_sizes) * mid_channels,
            mid_channels,
            3,
            padding=1, )
        self.cls = nn.Conv2D(mid_channels, num_classes, 1)

        self.fcn_head = nn.Sequential(
            layers.ConvBNReLU(
                self.backbone_dim + 256*4, mid_channels, 3, padding=1),
            nn.Conv2D(mid_channels, num_classes, 1), )  # self.feat_channels[2]

    def forward(self, x, target=None):
        feats = x[1]
        trans_feats = x[0]
        x = feats[-1]
        dcm_outs = [x]
        for dcm_module in self.dcm_modules:
            dcm_outs.append(dcm_module(x))
        dcm_outs = paddle.concat(dcm_outs, axis=1)
        x = self.bottleneck(dcm_outs)
        x = self.cls(x)
        x = F.interpolate(
            x, scale_factor=8, mode='bilinear', align_corners=True)
        output = [x]
        if self.training:
            trans_cnn_feat = paddle.concat([feats[2], trans_feats[1]], axis=1)
            # fcn_out = self.fcn_head(feats[2])
            fcn_out = self.fcn_head(trans_cnn_feat)
            fcn_out = F.interpolate(
                fcn_out, scale_factor=8, mode='bilinear', align_corners=True)
            output.append(fcn_out)

            labels = target['label'].astype('int64')
            
            loss = dict()
            seg_loss = self.loss(output, labels)
            loss['dmnetloss'] = seg_loss
            
            return loss

        return output


class DCM(nn.Layer):
    """
    Dynamic Convolutional Module used in DMNet.

    Args:
        filter_size (int): The filter size of generated convolution kernel used in Dynamic Convolutional Module.
        fusion (bool): Add one conv to fuse DCM output feature.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
    """

    def __init__(self, filter_size, fusion, in_channels, channels):
        super().__init__()
        self.filter_size = filter_size
        self.fusion = fusion
        self.channels = channels

        pad = (self.filter_size - 1) // 2
        if (self.filter_size - 1) % 2 == 0:
            self.pad = (pad, pad, pad, pad)
        else:
            self.pad = (pad + 1, pad, pad + 1, pad)

        self.avg_pool = nn.AdaptiveAvgPool2D(filter_size)
        self.filter_gen_conv = nn.Conv2D(in_channels, channels, 1)
        self.input_redu_conv = layers.ConvBNReLU(in_channels, channels, 1)

        self.norm = layers.SyncBatchNorm(channels)
        self.act = nn.ReLU()

        if self.fusion:
            self.fusion_conv = layers.ConvBNReLU(channels, channels, 1)

    def forward(self, x):
        generated_filter = self.filter_gen_conv(self.avg_pool(x))
        x = self.input_redu_conv(x)
        b, c, h, w = x.shape
        assert b > 0, "The batch size of x need to be bigger than 0, but got {}.".format(
            b)
        x = paddle.unsqueeze(
            paddle.flatten(
                x, start_axis=0, stop_axis=1), axis=0)
        generated_filter = generated_filter.reshape(
            [b * c, 1, self.filter_size, self.filter_size])

        x = F.pad(x, self.pad, mode='constant', value=0)
        output = F.conv2d(x, weight=generated_filter, groups=b * c)
        output = output.reshape([b, self.channels, h, w])
        output = self.norm(output)
        output = self.act(output)
        if self.fusion:
            output = self.fusion_conv(output)
        return output