'''
@Project ：transformer-master 
@File    ：position_wise_feed_forward.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/17 16:27 
'''
import torch.nn as nn
import torch.nn.functional as F
from multiheaded_attention import multiheaded_attention_testdata

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

def positionwise_feed_forward_testdata():
    """
    前馈全连接层测试数据制造
    :return:
    """
    d_model = 512
    # 线性变化的维度
    d_ff = 64
    dropout = 0.2
    x = multiheaded_attention_testdata()
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    ff_result = ff(x)
    return ff_result

if __name__ == '__main__':
    d_model = 512
    d_ff = 64
    dropout = 0.2
    x = multiheaded_attention_testdata()
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    ff_result = ff(x)
    print(f"ff_result:{ff_result}")
    print(f"shape of ff_result:{ff_result.shape}")