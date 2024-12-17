'''
@Project ：transformer-master 
@File    ：decoder.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/11/21 13:41 
'''
import torch.nn as nn
from repo.utils import clones
from layer_norm import LayerNorm

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)