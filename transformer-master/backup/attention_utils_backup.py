'''
@Project ：transformer-master 
@File    ：attention_utils.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/15 20:10 
'''
import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable
from bottom_to_up.embeddings import Embeddings
from bottom_to_up.positional_encoding import PositionalEncoding


def attention(query, key, value, mask=None, dropout=None):
    """注意力机制的实现， 输入分别是query,key,value,mask：掩码张量，
        dropout是nn.Dropout层的实例化对象，默认为None"""
    # 在函数中，首先取query的最后一维的大小，一般情况下就等同于我们的词嵌入维度，命名为d_k
    d_k = query.size(-1)
    # 按照注意力公式， 将query与key的转置相乘，这里面key是将最后两个维度进行转置,再除以缩放系数
    # 得到注意力得分张量scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 接着判断是否使用掩码张量
    if mask is not None:
        # 使用tensor的masked_fill方法，将掩码张量和scores张量每个位置一一比较，如果掩码张量
        # 则对应的scores张量用-1e9这个值来替换，如下演示
        scores = scores.masked_fill(mask == 0, -1e9)

    # 对scores的最后一维进行softmax操作，使用F.softmax方法，第一个参数是softmax对象，第二个
    # 这样获得最终的注意力张量
    p_attn = F.softmax(scores, dim=-1)
    # 之后判断是否使用dropout进行随机置0
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 最后，根据公式将p_attn与value张量相乘得最终的query注意力表示,同时返回注意力张量
    return torch.matmul(p_attn, value), p_attn

if __name__ == '__main__':
    d_model = 512
    dropout = 0.1
    max_len = 60

    d_model = 512
    # 词表大小是1000
    vocab = 1000
    # 输入x是一个使用Variable封装的长整型张量， 形状是2 × 4
    x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))

    emb = Embeddings(d_model, vocab)
    embr = emb(x)
    x = embr
    pe = PositionalEncoding(d_model, dropout, max_len)
    pe_result = pe(x)

    query = key = value = pe_result
    mask = Variable(torch.zeros(2, 4, 4))
    attn, p_attn = attention(query, key, value, mask=mask)
    print(f"attn:{attn}")
    print(f"shape of attn:{attn.shape}")
    print(f"p_attn:{p_attn}")
    print(f"shape of p_attn:{p_attn.shape}")