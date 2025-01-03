'''
@Project ：transformer-master 
@File    ：encoder_layer.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/11/20 23:00 
'''
import torch.nn as nn
from repo.utils import clones
from sublayer_connection import SublayerConnection
class EncoderLayer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
