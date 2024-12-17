'''
@Project ：transformer-master 
@File    ：positional_encoding.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/14 19:07 
'''
import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from embeddings import Embeddings
import matplotlib.pyplot as plt
import numpy as np
from embeddings import embeddings_testdata

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        """位置编码器类的初始化函数，共有三个参数，分别是d_model：词嵌入维度，
        dropout：置0比率， max_len：每个句子的最大长度"""
        super(PositionalEncoding, self).__init__()
        # 实例化nn中预定义的Dropout层，并将dropout传入其中，获得对象self.dropout
        self.dropout = nn.Dropout(p=dropout)
        # 初始化一个位置编码矩阵，它是一个全0矩阵，矩阵的大小是max_len × d_model
        pe = torch.zeros(max_len, d_model)
        # 初始化一个绝对位置矩阵，在我们这里，词汇的绝对位置就是用它的索引去表示。
        # 所以我们首先使用arange方法获得一个连续自然数向量，然后再使用unsqueeze方法拓展向理维度使其
        # 又因为参数传的是1,代表矩阵拓展的位置，会使向理变成一个max_len × 1 的矩阵。
        position = torch.arange(0, max_len).unsqueeze(1)

        # 绝对位置矩阵初始化之后，接下来就是考虑如何将这些位置信息加入到位置编码矩阵中，
        # 最简单思路就是先将 max_len × 1 的绝对位置矩阵，变换成max_len × d_model形状，然后覆盖原来的
        # 要做这种矩阵变换，就需要一个 1 × d_model 形状的变换矩阵div_term，我们对这个变换矩阵的要求除了
        # 还希望它能够将自然数的绝对位置编码缩放成足够小的数字，有助于在之后的梯度下阡过程中更快的收敛
        # 首先使用arange获得一个自然数矩阵，但是我们会发现，这里并没有按照预计的一样初
        # 而是有了一个跳跃，只初始化了一半即 1 × d_model / 2 的矩阵。为什么是一半呢，其实这里并不是真正
        # 我们可以把它看作是初始化了两次，而每次初始化的变换矩阵会做不同的处理，第一次初始化的变换
        # 并两个矩阵分别填充在位置编码矩阵的偶数和奇数位置上，组成最终的位置编码矩阵。
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 这样我们就得到了位置编码矩阵pe，pe现在还只是一个二维矩阵，要想和embedding的输出(一个三维张量) 相加，
        # 就必须拓展一个维度， 所以这里使用unsqueeze拓展维度。
        pe = pe.unsqueeze(0)

        # 最后把pe位置编码矩阵注册成模型的buffer，什么是buffer呢，
        # 我们把它认为是对模型效果有帮助的，但是却不是模型结构中超参数或者参数，不需要随着优化步骤进行更新的增益对象，
        # 注册之后我们就可以在模型保存后重加载时和模型结构与参数一同补加载。
        self.register_buffer('pe', pe)

    def forward(self, x):
        """forward函数的参数是x， 表示文本序列的词嵌入表示"""
        # 在相加之前我们对pe做一些适配工作， 将这个三维张量的第二维度也就是句子最大长度的那一维将切片到与输入的x的第二维相同
        # 因为我们默认max_len为5000一般来讲实在太大了，很难有一条句子包含5000个词汇，所以要进行与输入张量的适配，
        # 最后使用Variable进行封装，使其与x的样式相同，但是它是不需要进行梯度求解的，因此把requires.grad设置成false.
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        # 最后使用self.dropout对象进行'丢弃'操作，并返回结果。
        return self.dropout(x)


def show_pe():
    plt.figure(figsize=(15, 5))
    pe = PositionalEncoding(20, 0)
    y = pe(Variable(torch.zeros(1, 100, 20)))
    print(f"shape of y:{y.shape}")
    m = y[0, :, 4:8]
    print(f"m:{m}")
    print(f"shape of m:{m.shape}")
    plt.plot(np.arange(100), m.numpy())
    plt.legend(["dim %d" %p for p in [4,5,6,7]])
    plt.show()

def positional_encoding_testdata():
    """
    位置编码测试数据制造
    :return:pe_result:返回位置编码向量
    """
    emb_data = embeddings_testdata()
    x = emb_data

    dropout = 0.1
    max_len = 60
    d_model = 512
    pe = PositionalEncoding(d_model, dropout, max_len)
    pe_result = pe(x)
    return pe_result

if __name__ == '__main__':
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
    print(f"pe_result:{pe_result}")
    print(f"shape of pe_result:{pe_result.shape}")

    show_pe()