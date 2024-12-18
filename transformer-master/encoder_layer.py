'''
@Project ：transformer-master 
@File    ：encoder_layer.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/18 14:06 
'''
import torch
import torch.nn as nn
from utils import clones
from torch.autograd import Variable
from sublayer_connection import SublayerConnection
from multiheaded_attention import MultiHeadedAttention
from position_wise_feed_forward import PositionwiseFeedForward
from positional_encoding import positional_encoding_testdata
class EncoderLayer(nn.Module):
    """使用EncoderLayer类实现编码器层"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        """它的初始化函数参数有四个，分别是size，其实就是我们词嵌入维度的大小，它也将作为我们编码器层的大小,
           第二个self_attn，之后我们将传入多头自注意力子层实例化对象, 并且是自注意力机制,
           第三个是feed_froward, 之后我们将传入前馈全连接层实例化对象, 最后一个是置0比率dropout.
        """
        super(EncoderLayer, self).__init__()

        # 首先将self_attn和feed_forward传入其中.
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        # 如图所示, 编码器层中有两个子层连接结构, 所以使用clones函数进行克隆
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

        # 把size传入其中
        self.size = size

    def forward(self, x, mask):
        """forward函数中有两个输入参数，x和mask，分别代表上一层的输出，和掩码张量mask"""
        # 里面就是按照结构图左侧的流程. 首先通过第一个子层连接结构，其中包含多头自注意力子层，
        # 然后通过第二个子层连接结构，其中包含前馈全连接子层. 最后返回结果.
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

def encoder_layer_testdata():
    # 调用验证
    size = 512
    head = 8
    d_model = 512
    d_ff = 64
    pe_result = positional_encoding_testdata()
    x = pe_result
    dropout = 0.2
    self_attn = MultiHeadedAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    mask = Variable(torch.zeros(8, 4, 4))
    el = EncoderLayer(size, self_attn, ff, dropout)
    el_result = el(x, mask)
    return el_result

if __name__ == '__main__':
    # 调用验证
    size = 512
    head = 8
    d_model = 512
    d_ff = 64
    pe_result = positional_encoding_testdata()
    x = pe_result
    dropout = 0.2
    self_attn = MultiHeadedAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    mask = Variable(torch.zeros(8, 4, 4))
    el = EncoderLayer(size, self_attn, ff, dropout)
    el_result = el(x, mask)
    print(f"el_result:{el_result}")
    print(f"shape of el_result:{el_result.shape}")