'''
@Project ：transformer-master 
@File    ：position_wise_feed_forward.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/11/27 17:54 
'''
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    """
    Implements FFN equation.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))