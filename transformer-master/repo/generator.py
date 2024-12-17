'''
@Project ：transformer-master 
@File    ：generator.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/11/20 17:34 
'''
import torch.nn as nn
from torch.nn.functional import log_softmax

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

