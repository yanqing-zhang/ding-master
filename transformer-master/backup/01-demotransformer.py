# -*- coding: utf-8 -*-
# @Author  : Chinesejun
# @Email   : itcast@163.com
# @File    : 01-demotransformer.py
# @Software: PyCharm


# ===== transfromer架构解析 =====
import torch
import torch.nn as nn
import math
from torch.autograd import Variable

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model


    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

# embedding演示
# 8代表的是词表的大小, 也就是只能编码0-7
# 如果是10， 代表只能编码0-9  这里有11出现所以尚明embedding的时候要写成12
embedding = nn.Embedding(12, 3)
input = torch.LongTensor([[1, 2, 3, 4], [4, 11, 2, 1]])
# print('----:', embedding(input))

# padding_idx代表的意思就是和给定的值相同， 那么就用0填充， 比如idx=2那么第二行就是0
embedding = nn.Embedding(10, 3, padding_idx=5)
input = torch.LongTensor([[0,2,0,5]])
# print(embedding(input))
#
# 词嵌入维度是512维
d_model = 512

# 词表大小是1000
vocab = 1000
# 输入x是一个使用Variable封装的长整型张量, 形状是2 x 4 注意： 这里必须保证两句话的长度一致！！！
x = Variable(torch.LongTensor([[100,2,421,508],[491,998,1,221]]))
# x = torch.LongTensor([[100,2,421,508],[491,998,1,221]])
# print('x===', x)
emb = Embeddings(d_model, vocab)
embr = emb(x)
# print("embr:", embr)
print("embr:", embr.shape)

# 位置编码器
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout =nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        # max_lenX1形状
        positon = torch.arange(0, max_len).unsqueeze(1)
        # print("position===", positon.shape)
        # 1X (d_model/2)形状
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # print('div_term==', div_term)
        # print('div_term==', div_term.shape) #div_term== torch.Size([256])
        # print('sin部分==', torch.sin(positon*div_term))
        # print('sin部分==', torch.sin(positon*div_term).shape) # sin部分== torch.Size([60, 256])
        pe[:, 0::2] = torch.sin(positon*div_term)
        # print('cos部分==', torch.cos(positon*div_term))
        # print('cos部分==', torch.cos(positon*div_term).shape)# cos部分== torch.Size([60, 256])
        pe[:, 1::2] = torch.cos(positon*div_term)

        pe = pe.unsqueeze(0)
        '''
        向模块添加持久缓冲区。
        这通常用于注册不应被视为模型参数的缓冲区。例如，pe不是一个参数，而是持久状态的一部分。
        缓冲区可以使用给定的名称作为属性访问。

        说明：
        应该就是在内存中定义一个常量，同时，模型保存和加载的时候可以写入和读出
        '''
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print('x.size===', x.size(1))  # x.size=== 4
        # print('x.size===', x.shape)  # torch.Size([2, 4, 512])
        # print('pe.shape===', self.pe[:, :x.size(1)].shape)  # pe.shape=== torch.Size([1, 4, 512])
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

# m = nn.Dropout(p=0.2)
# input = torch.randn(4, 5)
# # 这里设置的是0.2, 但是随机的， 不一定就是4个
# print(m(input))

# x = torch.tensor([1, 2, 3, 4])
# # 在行的维度添加， 三维度添加之后， 形状是一样的
# print(torch.unsqueeze(x, 0))
# print(torch.unsqueeze(x, 0).shape)
# print(x.unsqueeze(0))
# print(x.unsqueeze(1))
# # 在列的维度添加  三纬度添加之后， 不管是写1还是0， 形状都是一样的
# print(torch.unsqueeze(x, 1))
# print(torch.unsqueeze(x, 1).shape)

# # 词嵌入维度是512维
# d_model = 512
#
# # 置0比率为0.1
dropout = 0.1
#
# # 句子最大长度
max_len=60
x = embr
#
pe = PositionalEncoding(d_model, dropout, max_len)
pe_result = pe(x)
# # print(pe_result)
# print(pe_result.shape) # torch.Size([2, 4, 512])
# #
import matplotlib.pyplot as plt
import numpy as np

# plt.figure(figsize=(15, 5))
# pe = PositionalEncoding(20, 0)
# y = pe(Variable(torch.zeros(1, 100, 20)))
# plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
# plt.legend(["dim %d"%p for p in [4,5,6,7]])
# plt.show()

# # 编码器部分
#
# '''
# [[[0. 1. 1. 1. 1.]
#   [0. 0. 1. 1. 1.]
#   [0. 0. 0. 1. 1.]
#   [0. 0. 0. 0. 1.]
#   [0. 0. 0. 0. 0.]]]
# '''
def subsequent_mask(size):
    attn_shape = (1, size, size)
    print('====', np.triu(np.ones(attn_shape), k=1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(1-subsequent_mask)

# # def triu（m, k）
# # m：表示一个矩阵
# # K：表示对角线的起始位置（k取值默认为0）
# # return: 返回函数的上三角矩阵
# # print(np.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], k=1))
# # print(np.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], k=0))
# # print(np.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], k=-1))
#
# # 调用验证
# size=5
# sm = subsequent_mask(size)
#
# print(sm)
# # print(subsequent_mask(20))
# # print(subsequent_mask(20)[0].shape) #torch.Size([20, 20])
# # plt.figure(figsize=(5, 5))
# # plt.imshow(subsequent_mask(20)[0])
# # plt.show()
#
# 注意力机制
import torch.nn.functional as F

def attention(query, key, value, mask=None, dropout=None):
    # print('query===', query)
    # print('queryshape===', query.shape) #queryshape=== torch.Size([2, 4, 512]) 2表示有2个样本， 4表示每个样本中四个词，512表示把每个词映射到512维度上(可以理解为512个特征)
    # print('querysize(-1)===', query.size(-1))  #querysize(-1)=== 512
    d_k = query.size(-1)
    # print('keyshape====', key.shape) # keyshape==== torch.Size([2, 4, 512])
    # print('keytranspose====', key.transpose(-2, -1).shape) # keytranspose==== torch.Size([2, 512, 4])
    # 这里的d_k为什么是词嵌入的维度，为什么要除以词嵌入的维度
    scores = torch.matmul(query, key.transpose(-2, -1)/math.sqrt(d_k))
    # print('scores+++++', scores)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # print('scoresmask-----', scores)
    # print('scoresmaskshape-----', scores.shape) #多头的:scoresmaskshape----- torch.Size([2, 8, 4, 4])
    '''
    scoresmask-----
    tensor([[[-1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09],
             [-1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09],
             [-1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09],
             [-1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09]],

            [[-1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09],
             [-1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09],
             [-1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09],
             [-1.0000e+09, -1.0000e+09, -1.0000e+09, -1.0000e+09]]]
    '''
    # print('scoreshape', scores.shape) #torch.Size([2, 4, 4])
    # 为什么对最后一维度进行softmax
    p_attn = F.softmax(scores, dim=-1)
    # print('p_attnsoftmax===', p_attn) p_attn= [2, 4, 4]
    '''
    添加mask掩码之后的
    tensor([[[0.2500, 0.2500, 0.2500, 0.2500],
             [0.2500, 0.2500, 0.2500, 0.2500],
             [0.2500, 0.2500, 0.2500, 0.2500],
             [0.2500, 0.2500, 0.2500, 0.2500]],

            [[0.2500, 0.2500, 0.2500, 0.2500],
             [0.2500, 0.2500, 0.2500, 0.2500],
             [0.2500, 0.2500, 0.2500, 0.2500],
             [0.2500, 0.2500, 0.2500, 0.2500]]]
    '''
    '''
    p_attnsoftmax=== 未添加mask掩码的
    tensor([[[1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0., 0., 0., 1.]],

            [[1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0., 0., 0., 1.]]]
    '''

    if dropout is not None:
        p_attn = dropout(p_attn)
        # print('p_attnshape', p_attn.shape)
                         # [2, 4, 4] X [2, 4, 512]=[2, 4, 512]
    return torch.matmul(p_attn, value), p_attn
#
# # 测试
# # input = Variable(torch.randn(5, 5))
# # mask = Variable(torch.zeros(5, 5))
# # print(input.masked_fill(mask==0, -1e9))
# # ====================
#
# query = key = value = pe_result # torch.Size([2, 4, 512])
# attn, p_attn = attention(query, key, value)
# # print('attn==', attn.shape) #attn== torch.Size([2, 4, 512])
# # print('p_attn==',p_attn.shape) # p_attn== torch.Size([2, 4, 4])
#
# mask = Variable(torch.zeros(2, 4, 4))
# attn, p_attn = attention(query, key, value, mask=mask)
# print("attn:", attn.shape) # attn: torch.Size([2, 4, 512])
# print("p_attn:", p_attn.shape) #p_attn== torch.Size([2, 4, 4])
#
# 多头注意力机制
import copy


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert embedding_dim%head== 0
        self.d_k = embedding_dim // head
        self.head = head
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value,  mask=None):

        if mask is not None:
            mask = mask.unsqueeze(0)
        # print('multmaskshape===', mask.shape) #multmaskshape=== torch.Size([1, 8, 4, 4])
        batch_size = query.size(0)
        # view中的四个参数的意义
        # batch_size: 批次的样本数量
        # -1这个位置应该是： 每个句子的长度
        # self.head*self.d_k应该是embedding的维度， 这里把词嵌入的维度分到了每个头中， 即每个头中分到了词的部分维度的特征
        # query, key, value形状torch.Size([2, 8, 4, 64])
        query, key, value = [model(x).view(batch_size, -1,  self.head, self.d_k).transpose(1, 2) for model, x in zip(self.linears, (query, key, value))]
        # query, key, value = [model(x) for model, x in zip(self.linears, (query, key, value))]
        # print('-=-=', query.shape)
        # print('-=-=', key.shape)
        # print('-=-=', value.shape)
        '''
        -=-= torch.Size([2, 4, 512])
        -=-= torch.Size([2, 4, 512])
        -=-= torch.Size([2, 4, 512])
        '''
        # 所以mask的形状 torch.Size([1, 8, 4, 4])  这里的所有参数都是4维度的   进过dropout的也是4维度的
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # contiguous解释:https://zhuanlan.zhihu.com/p/64551412
        # 这里相当于图中concat过程
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head*self.d_k)

        return self.linears[-1](x)
print("---------------------------------------")

# # view演示
# # x = torch.randn(4, 4)
# # print(x)
# #
# # y = x.view(16)
# # print(y)
#
# # a = torch.randn(1, 2, 3, 4)
# # print(a)
# # print(a.size())
# #
# # b = a.transpose(1, 2)
# # print(b)
# # print(b.size())
# #
# # # view是重新排成一排然后将其组合成要的形状
# # c = a.view(1, 3, 2, 4)
# # print(c)
# # print(c.size())
# #
#
#
# # 调用验证
# # 头数head
head = 8
#
# # 词嵌入维度embedding_dim
embedding_dim = 512
#
# # 置零比率dropout
dropout = 0.2
# # 假设输入的Q，K，V仍然相等
query = value = key = pe_result
#
# # 输入的掩码张量mask
mask = Variable(torch.zeros(8, 4, 4))
mha = MultiHeadedAttention(head, embedding_dim, dropout)
mha_result = mha(query, key, value, mask)
# print(mha_result)
print(mha_result.shape)
#
#

