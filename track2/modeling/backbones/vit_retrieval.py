from collections import OrderedDict
from typing import Tuple, Union

import math
import numpy as np
import paddle
import paddle.nn as nn
from paddle.nn import Linear
from paddle import ParamAttr
import paddle.nn.functional as F
from paddle.nn.initializer import Uniform, Constant, Normal

from .utils import init
# from .builder import BACKBONES
# from .base_transformer import AttentionPool2D
from .vision_transformer_v2 import Transformer, VisionTransformer


class LayerNorm(nn.LayerNorm):
    """Subclass Paddle's LayerNorm to handle fp16."""

    def forward(self, x):
        orig_type = x.dtype
        ret = super().forward(x.astype("float32"))
        return ret.astype(orig_type)


class QuickGELU(nn.Layer):

    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class CLIP(nn.Layer):

    def __init__(
            self,
            embed_dim,
            # vision
            image_resolution,
            vision_layers,
            vision_width,
            vision_patch_size,
            pre_norm,
            proj,
            patch_bias,
            # text
            context_length,
            vocab_size,
            transformer_width,
            transformer_heads,
            transformer_layers,
            qkv_bias):
        super().__init__()

        self.context_length = context_length

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            img_size=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            out_dim=embed_dim,
            depth=vision_layers,
            num_heads=vision_heads,
            pre_norm=pre_norm,
            proj=proj,
            patch_bias=patch_bias,
        )

        self.transformer = Transformer(
            embed_dim=transformer_width,
            depth=transformer_layers,
            num_heads=transformer_heads,
            attn_mask=self.build_attention_mask(context_length))

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)

        self.positional_embedding = self.create_parameter(
            shape=(self.context_length, transformer_width),
            default_initializer=Normal(std=0.01))
        self.add_parameter("positional_embedding", self.positional_embedding)

        self.ln_final = LayerNorm(transformer_width)
        scale = transformer_width**-0.5
        self.text_projection = self.create_parameter(
            shape=(transformer_width, embed_dim),
            default_initializer=Normal(std=scale))
        self.add_parameter("text_projection", self.text_projection)

        logit_ = Constant(value=np.log(1 / 0.07))
        self.logit_scale = self.create_parameter(shape=(1, ),
                                                 default_initializer=logit_)
        # self.logit_scale.stop_gradient = True
        self.add_parameter("logit_scale", self.logit_scale)

        self.initialize_parameters()

    def initialize_parameters(self):
        init.normal_init(self.token_embedding, std=0.02)
        init.normal_init(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.embed_dim**-0.5) * (
            (2 * self.transformer.depth))
        attn_std = self.transformer.embed_dim**-0.5
        fc_std = (2 * self.transformer.embed_dim)**-0.5
        for block in self.transformer.blocks:
            init.normal_init(block.attn.proj, std=proj_std)
            init.normal_init(block.attn.qkv, std=attn_std)
            init.normal_init(block.mlp.fc1, std=fc_std)
            init.normal_init(block.mlp.fc2, std=proj_std)

    def build_attention_mask(self, length):
        return paddle.tensor.triu((paddle.ones(
            (length, length), dtype=paddle.get_default_dtype()) * -np.inf), 1)

    @property
    def dtype(self):
        return self.visual.patch_embed.proj.weight.dtype

    def encode_image(self, image):
        return self.visual(image.astype("float32"))

    def encode_text(self, text):
        x = self.token_embedding(text).astype("float32")  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.astype("float32")
        x = self.transformer(x)
        # print('x after transformer', x.shape) # [28, 77, 512]
        x = self.ln_final(x).astype("float32")

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        idx = text.argmax(axis=-1)
        ran = paddle.arange(x.shape[0])
        x = paddle.concat([paddle.unsqueeze(x[i][idx[i]], axis=0) for i in ran],
                          axis=0)
        x = paddle.matmul(x, self.text_projection)

        return x

    def clip_logit_scale(self):
        logit_scale_buffer = self.logit_scale.clip_(-4.6, 4.6)
        logit_scale_buffer._share_buffer_to(self.logit_scale)

    def forward(self, inputs):
        is_train = self.training
        image = inputs['image']
        text = inputs['text']
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        # print('image_features', image_features.shape)
        # print('text_features', text_features.shape)

        # normalized features
        image_features = image_features / image_features.norm(axis=-1,
                                                              keepdim=True)
        text_features = text_features / text_features.norm(axis=-1,
                                                           keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp() if is_train else 1
        image_logits = paddle.matmul(logit_scale * image_features,
                                     text_features.t())
        text_logits = paddle.matmul(logit_scale * text_features,
                                    image_features.t())
        # self.clip_logit_scale()
        paddle.clip(self.logit_scale, -4.6, 4.6)


        return image_logits, text_logits
