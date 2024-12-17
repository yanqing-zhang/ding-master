'''
@Project ：transformer-master 
@File    ：utils.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/11/21 13:43 
'''
import torch
import torch.nn as nn
import copy
import math

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape),  diagonal=1).type(torch.uint8)
    return subsequent_mask == 0

def attention(query, key, value, mask=None, dropout=None):
    """
    注意力函数，实现缩放点积计算
    :param query:
    :param key:
    :param value:
    :param mask:
    :param dropout:
    :return:
    """
    # 获取 query 向量的最后一个维度的大小，通常这是键值对的维度，用于后续的计算。
    d_k = query.size(-1)
    # 计算 query 和 key 的转置（key.transpose(-2, -1)）之间的矩阵乘法，然后将每个分数除以根号d_k，这是缩放点积注意力的一个关键步骤，有助于在深度网络中稳定梯度。
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # 如果提供了 mask，这个 mask 用于屏蔽某些位置的注意力分数，使得这些位置在后续的softmax操作中几乎不会得到任何权重。这里，如果 mask 中的元素为0，则对应的 scores 位置会被填充为 -1e9（一个非常小的数），这样在softmax之后这些位置的权重将接近0。
        scores = scores.masked_fill(mask == 0, -1e9)
    # 对 scores 应用softmax函数，得到概率分布 p_attn。dim=-1 表示softmax操作是在最后一个维度上进行的。
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        # 如果提供了 dropout 参数，则对 p_attn 应用dropout操作。Dropout是一种正则化技术，可以防止模型过拟合。
        p_attn = dropout(p_attn)
    # 返回两个值：第一个是 p_attn 和 value 的矩阵乘法结果，这是经过注意力加权后的值；第二个是概率分布 p_attn 本身，它可能对模型的解释或调试有用。
    return torch.matmul(p_attn, value), p_attn