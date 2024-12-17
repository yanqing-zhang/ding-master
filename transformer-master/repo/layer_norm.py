'''
@Project ：transformer-master 
@File    ：layer_norm.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/11/20 17:55 
'''
import torch
import torch.nn as nn
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(num_features))
        self.b_2 = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2