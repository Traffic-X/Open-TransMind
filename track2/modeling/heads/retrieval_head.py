from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import paddle
import paddle.nn as nn


class RetrievalHead(nn.Layer):
    def __init__(self):
        super(RetrievalHead, self).__init__()

        self.criterion= nn.CrossEntropyLoss()


    def forward(self, inputs, targets=None):

        img_logits = inputs[0]
        text_logits = inputs[1]

        bs = img_logits.shape[0]
        img_labels = paddle.to_tensor(list(range(bs)), stop_gradient=True) #list(range(bs))
        text_labels = paddle.to_tensor(list(range(bs)), stop_gradient=True)

        img_loss = self.criterion(img_logits, img_labels)
        text_loss = self.criterion(text_logits, text_labels)

        loss = img_loss + text_loss
        outputs = {}
        outputs['loss_retrieval_img'] = img_loss
        outputs['loss_retrieval_text'] = text_loss
        outputs['loss_retrieval'] = loss 

        if self.training:
            return outputs
        else:
            return outputs