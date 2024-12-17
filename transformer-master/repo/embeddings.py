'''
@Project ：transformer-master 
@File    ：embeddings.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/11/27 18:16 
'''
import torch.nn as nn
import math
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)