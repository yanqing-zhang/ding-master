'''
@Project ：transformer-master 
@File    ：position_wise_feed_forward.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/17 16:27 
'''
import torch.nn as nn
import torch.nn.functional as F
class PositionwiseFeedForward(nn.Module):
    # 通过类PositionwiseFeedForward来实现前馈全连接层
    def __init__(self, d_model, d_ff, dropout=0.1):
        """初始化函数有三个输入参数分别是d_model, d_ff,和dropout=0.1，第一个是线性层的输入维度也是第二个线性层的输出维度，
                   因为我们希望输入通过前馈全连接层后输入和输出的维度不变. 第二个参数d_ff就是第二个线性层的输入维度和第一个线性层的输出维度.
                   最后一个是dropout置0比率."""
        super(PositionwiseFeedForward, self).__init__()

        # 首先按照我们预期使用nn实例化了两个线性层对象，self.w1和self.w2
        # 它们的参数分别是d_model, d_ff和d_ff, d_model
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        # 然后使用nn的Dropout实例化了对象self.dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """输入参数为x，代表来自上一层的输出"""
        # 首先经过第一个线性层，然后使用Funtional中relu函数进行激活,
        # 之后再使用dropout进行随机置0，最后通过第二个线性层w2，返回最终结果.
        return self.w2(self.dropout(F.relu(self.w1(x))))

if __name__ == '__main__':
    pass