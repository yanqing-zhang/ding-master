'''
@Project ：transformer-master 
@File    ：decoder.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/18 15:23 
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import clones
import copy
from layer_norm import LayerNorm
from position_wise_feed_forward import PositionwiseFeedForward
from multiheaded_attention import MultiHeadedAttention
from decoder_layer import DecoderLayer
from positional_encoding import positional_encoding_testdata
from encoder import encoder_testdata
class Decoder(nn.Module):
    """使用类Decoder来实现解码器"""
    def __init__(self, layer, N):
        """初始化函数的参数有两个，第一个就是解码器层layer，第二个是解码器层的个数N."""
        super(Decoder, self).__init__()

        # 首先使用clones方法克隆了N个layer，然后实例化了一个规范化层.
        # 因为数据走过了所有的解码器层后最后要做规范化处理.
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        """forward函数中的参数有4个，x代表目标数据的嵌入表示，memory是编码器层的输出，
           source_mask, target_mask代表源数据和目标数据的掩码张量
        """
        # 然后就是对每个层进行循环，当然这个循环就是变量x通过每一个层的处理，
        # 得出最后的结果，再进行一次规范化返回即可.
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


def decoder_testdata():
    # 调用验证
    # 分别是解码器层layer和解码器层的个数N
    size = 512
    d_model = 512
    head = 8
    d_ff = 64
    dropout = 0.2
    c = copy.deepcopy
    attn = MultiHeadedAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
    N = 8
    # 输入参数与解码器层的输入参数相同
    pe_result = positional_encoding_testdata()
    x = pe_result
    en_result = encoder_testdata()
    memory = en_result
    mask = Variable(torch.zeros(8, 4, 4))
    source_mask = target_mask = mask
    de = Decoder(layer, N)
    de_result = de(x, memory, source_mask, target_mask)
    return de_result


if __name__ == '__main__':
    # 调用验证
    # 分别是解码器层layer和解码器层的个数N
    size = 512
    d_model = 512
    head = 8
    d_ff = 64
    dropout = 0.2
    c = copy.deepcopy
    attn = MultiHeadedAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
    N = 8
    # 输入参数与解码器层的输入参数相同
    pe_result = positional_encoding_testdata()
    x = pe_result
    en_result = encoder_testdata()
    memory = en_result
    mask = Variable(torch.zeros(8, 4, 4))
    source_mask = target_mask = mask
    de = Decoder(layer, N)
    de_result = de(x, memory, source_mask, target_mask)
    print(f"de_result:{de_result}")
    print(f"shape of de_result:{de_result.shape}")