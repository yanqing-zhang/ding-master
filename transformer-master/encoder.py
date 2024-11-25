'''
@Project ：transformer-master 
@File    ：encoder.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/11/20 17:47 
'''
import torch.nn as nn
from utils import clones
from layer_norm import LayerNorm
class Encoder(nn.Module):

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)