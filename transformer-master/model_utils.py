'''
@Project ：transformer-master 
@File    ：model_utils.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/19 10:33 
'''
import copy
import torch.nn as nn
from multiheaded_attention import MultiHeadedAttention
from position_wise_feed_forward import PositionwiseFeedForward
from positional_encoding import PositionalEncoding
from encoder_decoder import EncoderDecoder
from encoder import Encoder
from encoder_layer import EncoderLayer
from decoder import Decoder
from decoder_layer import DecoderLayer
from generate import Generator
from embeddings import Embeddings

def make_model(source_vocab, target_vocab, N=6,
               d_model=512, d_ff=2048, head=8, dropout=0.1):
    """该函数用来构建模型, 有7个参数，分别是源数据特征(词汇)总数，目标数据特征(词汇)总数，
       编码器和解码器堆叠数，词向量映射维度，前馈全连接网络中变换矩阵的维度，
       多头注意力结构中的多头数，以及置零比率dropout.
    """
    # 首先得到一个深度拷贝命令，接下来很多结构都需要进行深度拷贝，
    # 来保证他们彼此之间相互独立，不受干扰.
    c = copy.deepcopy

    # 实例化了多头注意力类，得到对象attn
    attn = MultiHeadedAttention(head, d_model)

    # 然后实例化前馈全连接类，得到对象ff
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # 实例化位置编码类，得到对象position
    position = PositionalEncoding(d_model, dropout)

    # 根据结构图, 最外层是EncoderDecoder，在EncoderDecoder中，
    # 分别是编码器层，解码器层，源数据Embedding层和位置编码组成的有序结构，
    # 目标数据Embedding层和位置编码组成的有序结构，以及类别生成器层.
    # 在编码器层中有attention子层以及前馈全连接子层，
    # 在解码器层中有两个attention子层以及前馈全连接层.
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab)
    )

    # 模型结构完成后，接下来就是初始化模型中的参数，比如线性层中的变换矩阵
    # 这里一但判断参数的维度大于1，则会将其初始化成一个服从均匀分布的矩阵，
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


if __name__ == '__main__':
    source_vocab = 11
    target_vocab = 11
    N = 6

    res = make_model(source_vocab, target_vocab, N)
    print(f"res:{res}")
