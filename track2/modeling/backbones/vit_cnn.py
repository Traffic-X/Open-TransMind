# !/usr/bin/env python3

import math
import paddle
from functools import partial
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.distributed.fleet.utils import recompute

from paddleseg.models import layers
from paddleseg.utils import utils

from modeling.backbones.vision_transformer import drop_path, to_2tuple, trunc_normal_, zeros_, ones_
from ppdet.modeling.shape_spec import ShapeSpec


def load_checkpoint(model, pretrained):
    print('----- LOAD -----', pretrained)
    state_dict = paddle.load(pretrained)

    if 'pos_embed' in state_dict:
        print('---- POS_EMBED -----')
        pos_embed_checkpoint = state_dict['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        H, W = model.patch_embed.patch_shape
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = 1
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        print(orig_size, new_size)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            # extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape((-1, orig_size, orig_size, embedding_size)).transpose((0, 3, 1, 2))
            pos_tokens = F.interpolate(pos_tokens, size=(H, W), mode='bicubic', align_corners=False)
            new_pos_embed = pos_tokens.transpose((0, 2, 3, 1)).flatten(1, 2)
            # new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            state_dict['pos_embed'] = new_pos_embed

    model.set_state_dict(state_dict)
    print('----- LOAD END -----', pretrained)


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


class LastLevelMaxPool(nn.Layer):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return F.max_pool2d(x, kernel_size=1, stride=2, padding=0)


class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., lr=1.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, weight_attr=ParamAttr(learning_rate=lr), bias_attr=ParamAttr(learning_rate=lr))
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, weight_attr=ParamAttr(learning_rate=lr), bias_attr=ParamAttr(learning_rate=lr))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None, lr=1.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias_attr=qkv_bias, weight_attr=ParamAttr(learning_rate=lr))
        self.window_size = window_size
        assert len(self.window_size) == 2, "window_size must include two-dimension information"
        q_size = window_size[0]
        kv_size = q_size
        rel_sp_dim = 2 * q_size - 1

        self.rel_pos_h = self.create_parameter(
            shape=(2 * window_size[0] - 1, head_dim), default_initializer=zeros_,  attr=ParamAttr(learning_rate=lr))
        self.rel_pos_w = self.create_parameter(
            shape=(2 * window_size[1] - 1, head_dim), default_initializer=zeros_,  attr=ParamAttr(learning_rate=lr))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim, weight_attr=ParamAttr(learning_rate=lr), bias_attr=ParamAttr(learning_rate=lr))
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W, rel_pos_bias=None):
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape((B, N, 3, self.num_heads, -1)).transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]   # make paddlescript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q.matmul(k.transpose((0, 1, 3, 2)))
        # attn = calc_rel_pos_spatial(attn, q, self.window_size, self.window_size, self.rel_pos_h, self.rel_pos_w)
        _, _, q_length, _ = q.shape
        q_size = int(q_length ** 0.5)
        

        # if q_size == self.window_size[0]:
        if (H, W) == self.window_size:
            attn = calc_rel_pos_spatial(attn, q, self.window_size, self.window_size, self.rel_pos_h, self.rel_pos_w)
        else:
            rel_pos_h = F.interpolate(self.rel_pos_h.unsqueeze(axis=0), size=(2 * H - 1,), align_corners=False, mode='linear', data_format='NWC').squeeze(axis=0)
            rel_pos_w = F.interpolate(self.rel_pos_w.unsqueeze(axis=0), size=(2 * W - 1,), align_corners=False, mode='linear', data_format='NWC').squeeze(axis=0)
            attn = calc_rel_pos_spatial(attn, q, (H, W), (H, W), rel_pos_h, rel_pos_w)
        
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape((B, H // window_size, window_size, W // window_size, window_size, C))
    # windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    windows = x.transpose((0, 1, 3, 2, 4, 5)).reshape((-1, window_size, window_size, C))
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape((B, H // window_size, W // window_size, window_size, window_size, -1))
    x = x.transpose((0, 1, 3, 2, 4, 5)).reshape((B, H, W, -1))
    return x


def calc_rel_pos_spatial(
    attn,
    q,
    q_shape,
    k_shape,
    rel_pos_h,
    rel_pos_w,
    ):
    """
    Spatial Relative Positional Embeddings.
    """
    sp_idx = 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        paddle.arange(q_h)[:, None] * q_h_ratio - paddle.arange(k_h)[None, :] * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        paddle.arange(q_w)[:, None] * q_w_ratio - paddle.arange(k_w)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[paddle.to_tensor(dist_h, 'int64')]
    Rw = rel_pos_w[paddle.to_tensor(dist_w, 'int64')]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape((B, n_head, q_h, q_w, dim))
    rel_h = paddle.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = paddle.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].reshape((B, -1, q_h, q_w, k_h, k_w))
        + rel_h[:, :, :, :, :, None]
        + rel_w[:, :, :, :, None, :]
    ).reshape((B, -1, q_h * q_w, k_h * k_w))

    return attn


class WindowAttention(nn.Layer):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, 
        attn_drop=0., proj_drop=0., attn_head_dim=None, lr=1.0):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        q_size = window_size[0]
        kv_size = window_size[1]
        rel_sp_dim = 2 * q_size - 1

        self.rel_pos_h = self.create_parameter(
            shape=(rel_sp_dim, head_dim), default_initializer=zeros_, attr=ParamAttr(learning_rate=lr))
        self.rel_pos_w = self.create_parameter(
            shape=(rel_sp_dim, head_dim), default_initializer=zeros_, attr=ParamAttr(learning_rate=lr))

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias, weight_attr=ParamAttr(learning_rate=lr))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, weight_attr=ParamAttr(learning_rate=lr), bias_attr=ParamAttr(learning_rate=lr))
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        x = x.reshape((B_, H, W, C))
        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]

        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b), data_format='NHWC')
        _, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size[0])  # nW*B, window_size, window_size, C
        x = x.reshape((-1, self.window_size[1] * self.window_size[0], C))  # nW*B, window_size*window_size, C
        B_w = x.shape[0]
        N_w = x.shape[1]
        # qkv = self.qkv(x).reshape(B_w, N_w, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = self.qkv(x).reshape((B_w, N_w, 3, self.num_heads, C // self.num_heads)).transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]  # make paddlescript happy (cannot use tensor as tuple)
        
        q = q * self.scale
        # attn = (q @ k.transpose(-2, -1))
        attn = q.matmul(k.transpose((0, 1, 3, 2)))

        attn = calc_rel_pos_spatial(attn, q, self.window_size, self.window_size, self.rel_pos_h, self.rel_pos_w)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((B_w, N_w, C))
        # x = (attn @ v).transpose(1, 2).reshape((B_w, N_w, C))
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.reshape((-1, self.window_size[1], self.window_size[0], C))
        x = window_reverse(x, self.window_size[0], Hp, Wp)  # B H' W' C

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]

        x = x.reshape((B_, H * W, C))

        return x

class Block(nn.Layer):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, window=False, lr=1.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if not window:
            self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim, lr=lr)
        else:
            self.attn = WindowAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim, lr=lr)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, lr=lr)

        if init_values is not None:
            # self.gamma_1 = nn.Parameter(init_values * paddle.ones((dim)),requires_grad=True)
            # self.gamma_2 = nn.Parameter(init_values * paddle.ones((dim)),requires_grad=True)

            self.gamma_1 = self.create_parameter(shape=(dim), default_initializer=ones_) * init_values
            self.gamma_2 = self.create_parameter(shape=(dim), default_initializer=ones_) * init_values
            
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, H, W):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, lr=1.0):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,
                    weight_attr=ParamAttr(learning_rate=lr), bias_attr=ParamAttr(learning_rate=lr))

    def forward(self, x, **kwargs):
        x = self.proj(x)
        _, _, Hp, Wp = x.shape
        x = x.flatten(2).transpose((0, 2, 1))
        return x, (Hp, Wp)


class HybridEmbed(nn.Layer):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Layer)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with paddle.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(paddle.zeros((1, in_chans, img_size[0], img_size[1])))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose((0, 2, 1))
        x = self.proj(x)
        return x


class Norm2d(nn.Layer):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, epsilon=1e-6)

    def forward(self, x):
        x = x.transpose((0, 2, 3, 1))
        x = self.ln(x)
        x = x.transpose((0, 3, 1, 2))
        return x

############## CNN #################################################################

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


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 dilation=1,
                 data_format='NCHW'):
        super(BottleneckBlock, self).__init__()

        self.data_format = data_format
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act='relu',
            data_format=data_format)

        self.dilation = dilation

        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            dilation=dilation,
            data_format=data_format)
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
            data_format=data_format)

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first or stride == 1 else True,
                data_format=data_format)

        self.shortcut = shortcut
        # NOTE: Use the wrap layer for quantization training
        self.add = layers.Add()
        self.relu = layers.Activation(act="relu")

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = self.add(short, conv2)
        y = self.relu(y)
        return y


class BasicBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dilation=1,
                 shortcut=True,
                 if_first=False,
                 data_format='NCHW'):
        super(BasicBlock, self).__init__()
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            act='relu',
            data_format=data_format)
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            dilation=dilation,
            act=None,
            data_format=data_format)

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first or stride == 1 else True,
                data_format=data_format)

        self.shortcut = shortcut
        self.dilation = dilation
        self.data_format = data_format
        self.add = layers.Add()
        self.relu = layers.Activation(act="relu")

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = self.add(short, conv1)
        y = self.relu(y)

        return y


class ResNet_vd(nn.Layer):
    """
    The ResNet_vd implementation based on PaddlePaddle.

    The original article refers to Jingdong
    Tong He, et, al. "Bag of Tricks for Image Classification with Convolutional Neural Networks"
    (https://arxiv.org/pdf/1812.01187.pdf).

    Args:
        layers (int, optional): The layers of ResNet_vd. The supported layers are (18, 34, 50, 101, 152, 200). Default: 50.
        output_stride (int, optional): The stride of output features compared to input images. It is 8 or 16. Default: 8.
        multi_grid (tuple|list, optional): The grid of stage4. Defult: (1, 1, 1).
        in_channels (int, optional): The channels of input image. Default: 3.
        pretrained (str, optional): The path of pretrained model.

    """

    def __init__(self,
                 layers=50,
                 output_stride=8,
                 multi_grid=(1, 1, 1),
                 in_channels=3,
                 pretrained=None,
                 data_format='NCHW'):
        super(ResNet_vd, self).__init__()

        self.data_format = data_format
        self.conv1_logit = None  # for gscnn shape stream
        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512,
                        1024] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        # for channels of four returned stages
        self.feat_channels = [c * 4 for c in num_filters
                              ] if layers >= 50 else num_filters

        dilation_dict = None
        if output_stride == 8:
            dilation_dict = {2: 2, 3: 4}
        elif output_stride == 16:
            dilation_dict = {3: 2}

        self.conv1_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            act='relu',
            data_format=data_format)
        self.conv1_2 = ConvBNLayer(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            act='relu',
            data_format=data_format)
        self.conv1_3 = ConvBNLayer(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            act='relu',
            data_format=data_format)
        self.pool2d_max = nn.MaxPool2D(
            kernel_size=3, stride=2, padding=1, data_format=data_format)

        # self.block_list = []
        self.stage_list = []
        if layers >= 50:
            for block in range(len(depth)):
                shortcut = False
                block_list = []
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)

                    ###############################################################################
                    # Add dilation rate for some segmentation tasks, if dilation_dict is not None.
                    dilation_rate = dilation_dict[
                        block] if dilation_dict and block in dilation_dict else 1

                    # Actually block here is 'stage', and i is 'block' in 'stage'
                    # At the stage 4, expand the the dilation_rate if given multi_grid
                    if block == 3:
                        dilation_rate = dilation_rate * multi_grid[i]
                    ###############################################################################

                    bottleneck_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BottleneckBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block] * 4,
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0 and
                            dilation_rate == 1 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            dilation=dilation_rate,
                            data_format=data_format))

                    block_list.append(bottleneck_block)
                    shortcut = True
                self.stage_list.append(block_list)
        else:
            for block in range(len(depth)):
                shortcut = False
                block_list = []
                for i in range(depth[block]):
                    dilation_rate = dilation_dict[block] \
                        if dilation_dict and block in dilation_dict else 1
                    if block == 3:
                        dilation_rate = dilation_rate * multi_grid[i]

                    basic_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BasicBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block],
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0 \
                                and dilation_rate == 1 else 1,
                            dilation=dilation_rate,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            data_format=data_format))
                    block_list.append(basic_block)
                    shortcut = True
                self.stage_list.append(block_list)

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        self.conv1_logit = y.clone()
        y = self.pool2d_max(y)

        # A feature list saves the output feature map of each stage.
        feat_list = []
        for stage in self.stage_list:
            i = 0
            for block in stage:
                y = block(y)
            feat_list.append(y)

        return feat_list

    def init_weight(self):
        utils.load_pretrained_model(self, self.pretrained)




class ViT_ResNet(nn.Layer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=None, init_values=None, use_checkpoint=False, 
                 use_abs_pos_emb=False, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 out_indices=[11], pretrained=None, out_shape=None, lr_mult=1.0, extra_level=0, grainity=4,
                 layers=50,
                 output_stride=8,
                 multi_grid=(1, 1, 1),
                 in_channels=3,
                 res_pretrained=None,
                 data_format='NCHW'):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, epsilon=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, lr=lr_mult)

        self.patch_size = patch_size
        self.out_indices = out_indices
        self.extra_level = extra_level

        self._out_shape = out_shape
        self._out_strides = [4, 8, 16, 32]

        if use_abs_pos_emb:
            num_patches = self.patch_embed.num_patches
            self.pos_embed = self.create_parameter(
                shape=(1, num_patches, embed_dim), default_initializer=zeros_, attr=ParamAttr(learning_rate=lr_mult))
            
            self.pos_embed_x = img_size // patch_size
            self.pos_embed_y = img_size // patch_size
        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_checkpoint = use_checkpoint
        interval = depth // grainity
        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, 
                lr=lr_mult,
                window_size=(14, 14) if ((i + 1) % interval != 0) else self.patch_embed.patch_shape, 
                window=((i + 1) % interval != 0))
            for i in range(depth)])

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed)

        self.norm = norm_layer(embed_dim)

        if self.patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.Conv2DTranspose(embed_dim, embed_dim, kernel_size=2, stride=2),
                Norm2d(embed_dim),
                nn.GELU(),
                nn.Conv2DTranspose(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            self.fpn2 = nn.Sequential(nn.Conv2DTranspose(embed_dim, embed_dim, kernel_size=2, stride=2))
            self.fpn3 = nn.Identity()
            self.fpn4 = nn.MaxPool2D(kernel_size=2, stride=2)
        elif self.patch_size == 8:
            self.fpn1 = nn.Sequential(
                nn.Conv2DTranspose(embed_dim, embed_dim, kernel_size=2, stride=2),
                Norm2d(embed_dim),
                nn.GELU(),
            )
            self.fpn2 = nn.Identity()
            self.fpn3 = nn.MaxPool2D(kernel_size=2, stride=2)
            self.fpn4 = nn.MaxPool2D(kernel_size=4, stride=4)
        else:
            raise NotImplementedError("your patch size {} is not supported yet.".format(self.patch_size))
        
        extra_level_module_list = []
        for i in range(self.extra_level):
            extra_level_module_list.append(LastLevelMaxPool())
        self.extra_level_module_list = nn.LayerList(extra_level_module_list)

        self.apply(self._init_weights)
        self.fix_init_weight()
        self.pretrained = pretrained
        self.init_weights()


        ############## CNN #################################################################

        self.data_format = data_format
        self.conv1_logit = None  # for gscnn shape stream
        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512,
                        1024] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        # for channels of four returned stages
        self.feat_channels = [c * 4 for c in num_filters
                              ] if layers >= 50 else num_filters

        dilation_dict = None
        if output_stride == 8:
            dilation_dict = {2: 2, 3: 4}
        elif output_stride == 16:
            dilation_dict = {3: 2}

        self.conv1_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            act='relu',
            data_format=data_format)
        self.conv1_2 = ConvBNLayer(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            act='relu',
            data_format=data_format)
        self.conv1_3 = ConvBNLayer(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            act='relu',
            data_format=data_format)
        self.pool2d_max = nn.MaxPool2D(
            kernel_size=3, stride=2, padding=1, data_format=data_format)

        # self.block_list = []
        self.stage_list = []
        if layers >= 50:
            for block in range(len(depth)):
                shortcut = False
                block_list = []
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)

                    ###############################################################################
                    # Add dilation rate for some segmentation tasks, if dilation_dict is not None.
                    dilation_rate = dilation_dict[
                        block] if dilation_dict and block in dilation_dict else 1

                    # Actually block here is 'stage', and i is 'block' in 'stage'
                    # At the stage 4, expand the the dilation_rate if given multi_grid
                    if block == 3:
                        dilation_rate = dilation_rate * multi_grid[i]
                    ###############################################################################

                    bottleneck_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BottleneckBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block] * 4,
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0 and
                            dilation_rate == 1 else 1,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            dilation=dilation_rate,
                            data_format=data_format))

                    block_list.append(bottleneck_block)
                    shortcut = True
                self.stage_list.append(block_list)
        else:
            for block in range(len(depth)):
                shortcut = False
                block_list = []
                for i in range(depth[block]):
                    dilation_rate = dilation_dict[block] \
                        if dilation_dict and block in dilation_dict else 1
                    if block == 3:
                        dilation_rate = dilation_rate * multi_grid[i]

                    basic_block = self.add_sublayer(
                        'bb_%d_%d' % (block, i),
                        BasicBlock(
                            in_channels=num_channels[block]
                            if i == 0 else num_filters[block],
                            out_channels=num_filters[block],
                            stride=2 if i == 0 and block != 0 \
                                and dilation_rate == 1 else 1,
                            dilation=dilation_rate,
                            shortcut=shortcut,
                            if_first=block == i == 0,
                            data_format=data_format))
                    block_list.append(basic_block)
                    shortcut = True
                self.stage_list.append(block_list)

        self.res_pretrained = res_pretrained
        self.cnn_init_weight()

    def cnn_init_weight(self):
        utils.load_pretrained_model(self, self.res_pretrained)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param = param / math.sqrt(2.0 * layer_id)

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight, layer_id + 1)
            rescale(layer.mlp.fc2.weight, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        pretrained = pretrained or self.pretrained
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                zeros_(m.bias)
                ones_(m.weight)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            print(f"load from {pretrained}")
            load_checkpoint(self, pretrained)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)

        if self.pos_embed is not None:
            _, num_patches, embedding_size = x.shape
            _, pos_num_patches, _ = self.pos_embed.shape
            # if num_patches != pos_num_patches:
            #     orig_size = int(pos_num_patches ** 0.5)
            #     target_size = int(num_patches ** 0.5)
            #     pos_tokens = self.pos_embed.reshape((-1, orig_size, orig_size, embedding_size)).transpose((0, 3, 1, 2))
            #     pos_tokens = F.interpolate(pos_tokens, size=(target_size, target_size), mode='bicubic', align_corners=False)
            #     new_pos_embed = pos_tokens.transpose((0, 2, 3, 1)).flatten(1, 2)
            #     x = x + new_pos_embed
            if (self.pos_embed_x, self.pos_embed_y) != (Hp, Wp):
                pos_tokens = self.pos_embed.reshape((-1, self.pos_embed_x, self.pos_embed_y, embedding_size)).transpose((0, 3, 1, 2))
                pos_tokens = F.interpolate(pos_tokens, size=(Hp, Wp), mode='bicubic', align_corners=False)
                new_pos_embed = pos_tokens.transpose((0, 2, 3, 1)).flatten(1, 2)
                x = x + new_pos_embed
            else:
                x = x + self.pos_embed
        x = self.pos_drop(x)

        features = []
        for i, blk in enumerate(self.blocks):
            blk.H = Hp
            blk.W = Wp
            if self.use_checkpoint:
                x = recompute(blk, x, Hp, Wp) #TODO add checkpointing of paddle
            else:
                x = blk(x, Hp, Wp)
        
        x = self.norm(x)
        
        xp = x.transpose((0, 2, 1)).reshape((B, -1, Hp, Wp))

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(ops)):
            if i in self.out_indices:
                features.append(ops[i](xp))
        p5_feature = features[-1]

        for i in range(self.extra_level):
            features.append(self.extra_level_module_list[i](p5_feature))

        return tuple(features)

    def forward_cnn(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        self.conv1_logit = y.clone()
        y = self.pool2d_max(y)

        # A feature list saves the output feature map of each stage.
        feat_list = []
        for stage in self.stage_list:
            for block in stage:
                y = block(y)
            feat_list.append(y)

        return feat_list
    
    def forward(self, inputs):
        x = inputs['image']
        x1 = self.forward_features(x)
        x2 = self.forward_cnn(x)
        x = [x1, x2]  #TODO fuse CNN feature Transformer feature
        return x



    @property
    def out_shape(self):
        return [
            # ShapeSpec(channels=ch, stride=s) for ch, s in zip(self._out_shape, self._out_strides)
            ShapeSpec(channels=self._out_shape[i], stride=self._out_strides[i]) for i in self.out_indices
        ]