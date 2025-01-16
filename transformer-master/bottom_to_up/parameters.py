'''
@Project ：transformer-master 
@File    ：parameters.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2025/1/13 16:22 
'''
class Parameters():

    def __init__(self):
        """
        Embeddings:
        - d_model 词嵌入的维度
        - vocab 词表的大小

        PositionalEncoding:
        - d_model: 词嵌入维度
        - dropout: 置0比率
        - max_len=5000: 每个句子的最大长度

        attention:
        - query 查询 - PositionalEncoding
        - key 键 - PositionalEncoding
        - value 值 - PositionalEncoding
        - mask 掩码
        - dropout 丢弃率

        MultiHeadedAttention:
        - head: 头数
        - embedding_dim: 词嵌入的维度
        - dropout: 丢弃率

        PositionwiseFeedForward:
        - d_model: 词嵌入维度
        - d_ff: 前馈全连接层的维度
        - dropout: 丢弃率

        subsequent_mask:
        - size: 掩码张量最后两个维度的大小

        LayerNorm:
        - features: 特征数
        - eps: 一个很小的数 1e-6

        EncoderLayer:
        - size: 词嵌入维度
        - self_attn: 自注意力 - MultiHeadedAttention
        - feed_forward: 前馈全连接层 - PositionwiseFeedForward
        - dropout: 丢弃率

        Encoder:
        - layer: 编码器层 - EncoderLayer
        - N: 编码器层的个数

        SublayerConnection:
        - size: 词嵌入维度
        - dropout: 丢弃率

        DecoderLayer:
        - size: 代表词嵌入的维度大小, 同时也代表解码器层的尺寸
        - self_attn: 多头自注意力对象，也就是说这个注意力机制需要Q=K=V - MultiHeadedAttention
        - src_attn: 多头注意力对象，这里Q!=K=V - MultiHeadedAttention
        - feed_forward: 前馈全连接层 - PositionwiseFeedForward
        - dropout: 丢弃率

        Decoder:
        - layer: 解码器层 - DecoderLayer
        - N: 解码器层的个数

        Generator:
        - d_model: 词嵌入维度
        - vocab_size: 词表大小

        EncoderDecoder:
        - encoder: 编码器 - Encoder
        - decoder: 解码器 - Decoder
        - src_embed: 源数据嵌入函数 - nn.Embedding
        - tgt_embed: 目标数据嵌入函数 - nn.Embedding
        - generator: 输出部分的类别生成器对象 - Generator

        clones:
        - module: 模型 - nn.Module
        - N: 模型的个数
        = ~ --------------------------------------------
        input-parameters:
        - d_model 词嵌入的维度 - 512
        - vocab 词表的大小 - 100
        - dropout 丢弃率 - 0.1
        - max_len=5000: 每个句子的最大长度
        - head: 头数 - 8
        - d_ff: 前馈全连接层的维度 - 1024
        - size: 掩码张量最后两个维度的大小
        - features: 特征数
        - eps: 一个很小的数 1e-6
        - N: 编码器层的个数 - 6
        """
