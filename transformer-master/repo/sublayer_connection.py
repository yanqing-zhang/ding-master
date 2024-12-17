'''
@Project ：transformer-master 
@File    ：sublayer_connection.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/11/20 18:15 
'''
import torch.nn as nn
from layer_norm import LayerNorm
class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))