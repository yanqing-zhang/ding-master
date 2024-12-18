'''
@Project ：transformer-master 
@File    ：encoder.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/18 14:26 
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import clones
import copy
from layer_norm import LayerNorm
from multiheaded_attention import MultiHeadedAttention
from position_wise_feed_forward import PositionwiseFeedForward
from encoder_layer import EncoderLayer
from positional_encoding import positional_encoding_testdata
class Encoder(nn.Module):
    """使用Encoder类来实现编码器"""
    def __init__(self, layer, N):
        super(Encoder, self).__init__()

        # 首先使用clones函数克隆N个编码器层放在self.layers中
        self.layers = clones(layer, N)

        # 再初始化一个规范化层, 它将用在编码器的最后面.
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """forward函数的输入和编码器层相同, x代表上一层的输出, mask代表掩码张量"""

        # 首先就是对我们克隆的编码器层进行循环，每次都会得到一个新的x，
        # 这个循环的过程，就相当于输出的x经过了N个编码器层的处理.
        # 最后再通过规范化层的对象self.norm进行处理，最后返回结果.
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

def encoder_testdata():
    # 调用验证
    # 第一个实例化参数layer, 它是一个编码器层的实例化对象, 因此需要传入编码器层的参数
    # 又因为编码器层中的子层是不共享的, 因此需要使用深度拷贝各个对象.
    size = 512
    head = 8
    d_model = 512
    d_ff = 64
    dropout = 0.2
    c = copy.deepcopy
    attn = MultiHeadedAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    layer = EncoderLayer(size, c(attn), c(ff), dropout)

    # 编码器中编码器层的个数N
    N = 8
    mask = Variable(torch.zeros(8, 4, 4))
    en = Encoder(layer, N)
    pe_result = positional_encoding_testdata()
    x = pe_result
    en_result = en(x, mask)
    return en_result

if __name__ == '__main__':
    # 调用验证
    # 第一个实例化参数layer, 它是一个编码器层的实例化对象, 因此需要传入编码器层的参数
    # 又因为编码器层中的子层是不共享的, 因此需要使用深度拷贝各个对象.
    size = 512
    head = 8
    d_model = 512
    d_ff = 64
    dropout = 0.2
    c = copy.deepcopy
    attn = MultiHeadedAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    layer = EncoderLayer(size, c(attn), c(ff), dropout)

    # 编码器中编码器层的个数N
    N = 8
    mask = Variable(torch.zeros(8, 4, 4))
    en = Encoder(layer, N)
    pe_result = positional_encoding_testdata()
    x = pe_result
    en_result = en(x, mask)
    print(f"en_result:{en_result}")
    print(f"shape of en_result:{en_result.shape}")