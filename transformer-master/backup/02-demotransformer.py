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
# dropout = 0.1
#
# # 句子最大长度
# max_len=60
# x = embr
#
# pe = PositionalEncoding(d_model, dropout, max_len)
# pe_result = pe(x)
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
# head = 8
#
# # 词嵌入维度embedding_dim
# embedding_dim = 512
#
# # 置零比率dropout
# dropout = 0.2
# # 假设输入的Q，K，V仍然相等
# query = value = key = pe_result
#
# # 输入的掩码张量mask
# mask = Variable(torch.zeros(8, 4, 4))
# mha = MultiHeadedAttention(head, embedding_dim, dropout)
# mha_result = mha(query, key, value, mask)
# # print(mha_result)
# print(mha_result.shape)
#
#
# ===== 前馈全连接层 =====

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


# # 调用验证
# d_model = 512
# d_ff = 64
# dropout = 0.2
# x = mha_result
# ff = PositionwiseFeedForward(d_model, d_ff, dropout)
# ff_result = ff(x)
# # print('ff_result==', ff_result)
# print('ff_resultshape==', ff_result.shape)
#
# ===== 规范化层 =====

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.a2 = nn.Parameter(torch.ones(features))
        # print('self.a2==', self.a2.shape) # self.a2== torch.Size([512])
        self.b2 = nn.Parameter(torch.zeros(features))
        # print('self.b2 = ', self.b2)
        self.eps = eps

    def forward(self, x):
        # keepdim资料： https://blog.csdn.net/qq_36810398/article/details/104845401
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # print('====',self.a2 * (x - mean))
        return self.a2 * (x - mean) / (std + self.eps) + self.b2

# features = d_model = 512
# eps = 1e-6
# x = ff_result
# ln = LayerNorm(features, eps)
# ln_result = ln(x)
# # print(ln_result)
# print(ln_result.shape)
#
# 子层链接结构
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
        # return x + self.dropout(self.norm(sublayer(x)))

# size = 512
# dropout = 0.2
# head = 8
# d_model = 512
# x = pe_result
# mask = Variable(torch.zeros(8, 4, 4))
# self_attn = MultiHeadedAttention(head, d_model)
# sublayer = lambda x:self_attn(x, x, x, mask)
#
# sc = SublayerConnection(size, dropout)
# sc_result = sc(x, sublayer)
# print(sc_result)
# print('sc_resultshape====', sc_result.shape)
#
# # 编码器层
# class EncoderLayer(nn.Module):
#     def __init__(self, size, self_attn, feed_forward, dropout):
#         super(EncoderLayer, self).__init__()
#         self.self_attn = self_attn
#         self.feed_forward = feed_forward
#         self.sublayer = clones(SublayerConnection(size, dropout), 2)
#         self.size = size
#
#     def forward(self, x, mask):
#         x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
#         return self.sublayer[1](x, self.feed_forward)
#
# size = 512
# head = 8
# d_model = 512
# d_ff = 64
# x = pe_result
# dropout = 0.2
# self_attn = MultiHeadedAttention(head, d_model)
# ff = PositionwiseFeedForward(d_model, d_ff, dropout)
# mask = Variable(torch.zeros(8, 4, 4))
#
# el = EncoderLayer(size, self_attn, ff, dropout)
# el_result = el(x, mask)
# print(el_result)
# print(el_result.shape)
#
#
# # 编码器
# class Encoder(nn.Module):
#     def __init__(self, layer, N):
#         super(Encoder, self).__init__()
#         self.layers = clones(layer, N)
#         self.norm = LayerNorm(layer.size)
#
#
#     def forward(self, x, mask):
#         for layer in self.layers:
#             x = layer(x, mask)
#
#         return self.norm(x)
#
# # 第一个实例化参数layer, 它是一个编码器层的实例化对象, 因此需要传入编码器层的参数
# # 又因为编码器层中的子层是不共享的, 因此需要使用深度拷贝各个对象.
# size = 512
# head = 8
# d_model = 512
# d_ff = 64
# c = copy.deepcopy
# attn = MultiHeadedAttention(head, d_model)
# ff = PositionwiseFeedForward(d_model, d_ff, dropout)
# dropout = 0.2
# layer = EncoderLayer(size, c(attn), c(ff), dropout)
#
# # 编码器中编码器层的个数N
# N = 8
# mask = Variable(torch.zeros(8, 4, 4))
#
# en = Encoder(layer, N)
# en_result = en(x, mask)
# # print(en_result)
# # print(en_result.shape)
#
# # 解码器
#
# class DecoderLayer(nn.Module):
#     def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
#         super(DecoderLayer, self).__init__()
#         self.size = size
#         self.self_attn = self_attn
#         self.src_attn = src_attn
#         self.feed_forward = feed_forward
#         self.sublayer = clones(SublayerConnection(size, dropout), 3)
#
#
#     def forward(self, x, memory, source_mask, target_mask):
#         m = memory
#         x = self.sublayer[0](x, lambda x:self.self_attn(x, x, x, target_mask))
#         x = self.sublayer[1](x, lambda x:self.src_attn(x, m, m, source_mask))
#         return self.sublayer[2](x, self.feed_forward)
#
# # 类的实例化参数与解码器层类似, 相比多出了src_attn, 但是和self_attn是同一个类.
# head = 8
# size = 512
# d_model = 512
# d_ff = 64
# dropout = 0.2
# self_attn = src_attn = MultiHeadedAttention(head, d_model, dropout)
#
# # 前馈全连接层也和之前相同
# ff = PositionwiseFeedForward(d_model, d_ff, dropout)
# x = pe_result
# memory = en_result
# mask = Variable(torch.zeros(8, 4, 4))
# source_mask = target_mask = mask
#
# dl = DecoderLayer(size, self_attn, src_attn, ff, dropout)
# dl_result = dl(x, memory, source_mask, target_mask)
# print(dl_result)
# print(dl_result.shape)

# # 解码器
# class Decoder(nn.Module):
#     def __init__(self, layer, N):
#         super(Decoder, self).__init__()
#         self.layers = clones(layer, N)
#         self.norm = LayerNorm(layer.size)
#
#
#
#     def forward(self, x, memory, source_mask, target_mask):
#         for layer in self.layers:
#             x = layer(x, memory,  source_mask, target_mask)
#         return self.norm(x)
#         pass
#
# # 分别是解码器层layer和解码器层的个数N
# size = 512
# d_model = 512
# head = 8
# d_ff = 64
# dropout = 0.2
# c = copy.deepcopy
# attn = MultiHeadedAttention(head, d_model)
# ff = PositionwiseFeedForward(d_model, d_ff, dropout)
# layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
# N = 8
# # 输入参数与解码器层的输入参数相同
# x = pe_result
# memory = en_result
# mask = Variable(torch.zeros(8, 4, 4))
# source_mask = target_mask = mask
# de = Decoder(layer, N)
# de_result = de(x, memory, source_mask, target_mask)
# # print(de_result)
# # print(de_result.shape)
#
#
# # 输出部分实现
# import torch.nn.functional as F
#
# class Generator(nn.Module):
#     def __init__(self, d_model, vocab_size):
#         super(Generator, self).__init__()
#         self.project = nn.Linear(d_model, vocab_size)
#
#
#     def forward(self, x):
#         return F.log_softmax(self.project(x), dim=-1)
#         pass
#
# # 词嵌入维度是512维
# d_model = 512
#
# # 词表大小是1000
# vocab_size = 1000
# # 输入x是上一层网络的输出, 我们使用来自解码器层的输出
# x = de_result
#
# gen = Generator(d_model, vocab_size)
# gen_result = gen(x)
# # print(gen_result)
# print('gen_resultshape===', gen_result.shape)
#
# 模型构建

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator



    def forward(self, source,  target, source_mask, target_mask):
        # 课件中缺少generator的使用
        return self.generator(self.decode(self.encode(source, source_mask), source_mask, target, target_mask))

    def encode(self, source, source_mask):
        return self.encoder(self.src_embed(source), source_mask)


    def decode(self, memory, souce_mask, target, target_mask):
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)

# vocab_size = 1000
# d_model = 512
# encoder = en
# decoder = de
# source_embed = nn.Embedding(vocab_size, d_model)
# target_embed = nn.Embedding(vocab_size, d_model)
# generator = gen
#
# # 假设源数据与目标数据相同, 实际中并不相同
# source = target = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
#
# # 假设src_mask与tgt_mask相同，实际中并不相同
# source_mask = target_mask = Variable(torch.zeros(8, 4, 4))
#
# ed = EncoderDecoder(encoder, decoder, source_embed, target_embed, generator)
# ed_result = ed(source, target, source_mask, target_mask)
# # print(ed_result)
# print('ed_result ==', ed_result.shape)
#
# # Transformer模型构建
# def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
#     c = copy.deepcopy
#
#     attn = MultiHeadedAttention(head, d_model)
#     ff = PositionwiseFeedForward(d_model, d_ff, dropout)
#     position = PositionalEncoding(d_model, dropout)
#
#     model = EncoderDecoder(
#         Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
#         Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
#         # nn.Sequential扩展资料：https://blog.csdn.net/qq_42518956/article/details/104662581
#         nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
#         nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
#         Generator(d_model, target_vocab)
#     )
#
#     for p in model.parameters():
#         if p.dim() > 1:
#             # nn.init.xavier_uniform扩展资料：https://blog.csdn.net/luoxuexiong/article/details/95772045
#             # nn.init.xavier_uniform(p)
#             nn.init.xavier_uniform_(p)
#
#     return model
#
# # nn.init.xavier_uniform演示
# # w = torch.empty(3, 5)
# # print(nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu')))
#
# source_vocab = 11
# target_vocab = 11
# N = 6
# # 其他参数都使用默认值
# if __name__ == '__main__':
#     res = make_model(source_vocab, target_vocab, N)
#     print(res)
#
# # ===== 模型的基础测试  =====
#
#
#
# # 第一步: 构建数据集生成器
# # 导入工具包Batch, 它能够对原始样本数据生成对应批次的掩码张量
# from pyitcast.transformer_utils import Batch, run_epoch
#
#
# def data_generator(V, batch, num_batch):
#     """该函数用于随机生成copy任务的数据, 它的三个输入参数是V: 随机生成数字的最大值+1,
#        batch: 每次输送给模型更新一次参数的数据量, num_batch: 一共输送num_batch次完成一轮
#     """
#     # 使用for循环遍历nbatches
#     for i in range(num_batch):
#         # 在循环中使用np的random.randint方法随机生成[1, V)的整数,
#         # 分布在(batch, 10)形状的矩阵中, 然后再把numpy形式转换称torch中的tensor.
#         data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
#
#         # 接着使数据矩阵中的第一列数字都为1, 这一列也就成为了起始标志列,
#         # 当解码器进行第一次解码的时候, 会使用起始标志列作为输入.
#         data[:, 0] = 1
#
#         # 因为是copy任务, 所有source与target是完全相同的, 且数据样本作用变量不需要求梯度
#         # 因此requires_grad设置为False
#         source = Variable(data, requires_grad=False)
#         target = Variable(data, requires_grad=False)
#
#         # 使用Batch对source和target进行对应批次的掩码张量生成, 最后使用yield返回
#         yield Batch(source, target)
# # 将生成0-10的整数
# V = 11
#
# # 每次喂给模型20个数据进行参数更新
# batch = 20
#
# # 连续喂30次完成全部数据的遍历, 也就是1轮
# num_batch = 30
#
# # if __name__ == '__main__':
# #     res = data_generator(V, batch, num_batch)
# #     print(res)
#
# # 导入优化器工具包get_std_opt, 该工具用于获得标准的针对Transformer模型的优化器
# # 该标准优化器基于Adam优化器, 使其对序列到序列的任务更有效.
# from pyitcast.transformer_utils import get_std_opt
#
# # 导入标签平滑工具包, 该工具用于标签平滑, 标签平滑的作用就是小幅度的改变原有标签值的值域
# # 因为在理论上即使是人工的标注数据也可能并非完全正确, 会受到一些外界因素的影响而产生一些微小的偏差
# # 因此使用标签平滑来弥补这种偏差, 减少模型对某一条规律的绝对认知, 以防止过拟合. 通过下面示例了解更多.
# from pyitcast.transformer_utils import LabelSmoothing
#
# # 导入损失计算工具包, 该工具能够使用标签平滑后的结果进行损失的计算,
# # 损失的计算方法可以认为是交叉熵损失函数.
# from pyitcast.transformer_utils import SimpleLossCompute
#
# # 使用make_model获得model
# model = make_model(V, V, N=2)
#
# # 使用get_std_opt获得模型优化器
# model_optimizer = get_std_opt(model)
#
# # 使用LabelSmoothing获得标签平滑对象
# criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
#
# # 使用SimpleLossCompute获得利用标签平滑结果的损失计算方法
# loss = SimpleLossCompute(model.generator, criterion, model_optimizer)
#
# from pyitcast.transformer_utils import LabelSmoothing
#
# # 使用LabelSmoothing实例化一个crit对象.
# # 第一个参数size代表目标数据的词汇总数, 也是模型最后一层得到张量的最后一维大小
# # 这里是5说明目标词汇总数是5个. 第二个参数padding_idx表示要将那些tensor中的数字
# # 替换成0, 一般padding_idx=0表示不进行替换. 第三个参数smoothing, 表示标签的平滑程度
# # 如原来标签的表示值为1, 则平滑后它的值域变为[1-smoothing, 1+smoothing].
# crit = LabelSmoothing(size=5, padding_idx=0, smoothing=0.5)
#
# # 假定一个任意的模型最后输出预测结果和真实结果
# predict = Variable(torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
#                              [0, 0.2, 0.7, 0.1, 0],
#                              [0, 0.2, 0.7, 0.1, 0]]))
# #
# # 标签的表示值是0，1，2
# target = Variable(torch.LongTensor([2, 1, 0]))
#
# # 将predict, target传入到对象中
# crit(predict, target)
#
# import matplotlib.pyplot as plt
# # 绘制标签平滑图像
# plt.imshow(crit.true_dist)
# plt.show()
#
# # 导入贪婪解码工具包greedy_decode, 该工具将对最终结进行贪婪解码
# # 贪婪解码的方式是每次预测都选择概率最大的结果作为输出,
# # 它不一定能获得全局最优性, 但却拥有最高的执行效率.
# from pyitcast.transformer_utils import greedy_decode
# from pyitcast.transformer_utils import run_epoch
#
# def run(model, loss, epochs=10):
#     for epoch in range(epochs):
#         model.train()
#
#         run_epoch(data_generator(V, 8, 20), model, loss)
#
#         model.eval()
#
#         run_epoch(data_generator(V, 8, 5), model, loss)
#
#     # 模型进入测试模式
#     model.eval()
#
#     # 假定的输入张量
#     source = Variable(torch.LongTensor([[1,3,2,5,4,6,7,8,9, 10]]))
#
#     # 定义源数据掩码张量, 因为元素都是1, 在我们这里1代表不遮掩
#     # 因此相当于对源数据没有任何遮掩.
#     # source_mask = Variable(torch.ones(1, 1, 10))
#
#     # 最后将model, src, src_mask, 解码的最大长度限制max_len, 默认为10
#     # 以及起始标志数字, 默认为1, 我们这里使用的也是1
#     result = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
#     print(result)


# if __name__ == '__main__':
#     run(model, loss) # 运行会报错
# from pyitcast.transformer import TransformerModel

# # ===== seq2seq模型架构实现英译法任务 =====
#
# # 第一步: 导入必备的工具包.
# from io import open
# import unicodedata
# import re
# import random
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import optim
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# # 第二步: 对持久化文件中数据进行处理, 以满足模型训练要求.
# SOS_token = 0
# EOS_token = 1
# class Lang(object):
#     def __init__(self, name):
#         self.name = name
#         self.word2index = {}
#         self.index2word = {0:'SOS', 1:'EOS'}
#         self.n_words = 2
#
#     def addSentence(self, sentence):
#         for word in sentence.split(' '):
#             self.addWord(word)
#
#     def addWord(self, word):
#         if word not in self.word2index:
#             self.word2index[word] = self.n_words
#             self.index2word[self.n_words] = word
#             self.n_words += 1
#
# # 调用验证
# name = 'eng'
# sentence = 'hello I am Jay'
#
# eng1 = Lang(name)
# eng1.addSentence(sentence)
# # print("word2index:", eng1.word2index)
# # print("index2word:", eng1.index2word)
# # print("n_words:", eng1.n_words)
#
# def unicodeToAscii(s):
#     return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
#
# def normalizeString(s):
#     s = unicodeToAscii(s.lower().strip())
#     s = re.sub(r"([.!?])", r" \1", s)
#     s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
#     return s
#
# # 调用验证
# # s = "Are you kidding me?"
# # nsr = normalizeString(s)
# # print(nsr)
#
# data_path = './data/eng-fra.txt'
# def readLangs(lang1, lang2):
#     # print('====', open(data_path, encoding='utf-8').read().strip())
#     lines = open(data_path, encoding='utf-8').read().strip().split('\n')
#     print("lines===:", lines)
#     "He looks pale. He must have drunk too much last night.\tIl n'a pas l'air bien. Il a dû trop boire la nuit dernière."
#     paris = [[normalizeString(s) for s in l.split('\t')]for l in lines]
#     input_lang = Lang(lang1)
#     output_lang = Lang(lang2)
#
#     return input_lang, output_lang, paris
# #调用验证
# lang1 = 'eng'
# lang2 = 'fra'
#
# input_lang, output_lang, pairs = readLangs(lang1, lang2)
# # print("input_lang:", input_lang)
# # print("output_lang:", output_lang)
# # print("pairs中的前五个:", pairs[:5])
#
# MAX_LENGTH = 10
#
# eng_prefixes = (
#     "i am ", "i m ",
#     "he is", "he s ",
#     "she is", "she s ",
#     "you are", "you re ",
#     "we are", "we re ",
#     "they are", "they re "
# )
#
# def filterPair(p):
#     return len(p[0].split(' ')) < MAX_LENGTH and p[0].startswith(eng_prefixes) and len(p[1].split(' ')) < MAX_LENGTH
#
# def filterPairs(pairs):
#     return [pair for pair in pairs if filterPair(pair)]
#
# # 调用验证
# fpairs = filterPairs(pairs)
# # print("过滤后的pairs前五个:", fpairs[:5])
#
# def prepareData(lang1, lang2):
#     input_lang, output_lang, pairs = readLangs(lang1, lang2)
#     pairs = filterPairs(pairs)
#     for pair in pairs:
#         input_lang.addSentence(pair[0])
#         output_lang.addSentence(pair[1])
#
#     return input_lang, output_lang, pairs
#
# # 调用验证
# input_lang, output_lang, pairs = prepareData('eng', 'fra')
# # print("input_n_words:", input_lang.n_words)
# # print("output_n_words:", output_lang.n_words)
# # print(random.choice(pairs))
#
# def tensorFromSentence(lang, sentence):
#     indexes = [lang.word2index[word] for word in sentence.split(' ')]
#     indexes.append(EOS_token)
#     return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)
#
# def tensorFromPair(pair):
#     input_tensor = tensorFromSentence(input_lang, pair[0])
#     target_tensor = tensorFromSentence(output_lang, pair[1])
#     return (input_tensor, target_tensor)
#
# # pairs形状： [['i m not ready .', 'je ne suis pas pret .'],...]
# pair = pairs[0]
# pair_tensor = tensorFromPair(pair)
# print(pair_tensor)
#
# # 第三步: 构建基于GRU的编码器和解码器.
#
# class EncoderRNN(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(EncoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         # self.embedding = nn.Embedding(input_size, hidden_size)
#         # 注意: 这里的15是原来的hidden_size是可以任意修改的， 和GRU第二个参数的hidden_size是不一样的， GRU的第二参数hidden_size必须是传入的参数hidden_size
#         # 这里的input_size是对应的词映射为对应张量的值 即input== tensor([2])， 这里是一个词对应的张量值， 所以后面尺寸才是[1, 15]
#         self.embedding = nn.Embedding(input_size, 15)
#         self.gru = nn.GRU(15, hidden_size)
#         # self.gru = nn.GRU(hidden_size, hidden_size)
#
#
#     def forward(self, input, hidden):
#         # output = self.embedding(input)
#         # print('embeddingshape===', output.shape) # torch.Size([1, 15])
#         output = self.embedding(input).view(1, 1, -1)
#         # print('outputshape===', output.shape) #torch.Size([1, 1, 15])
#         output, hidden = self.gru(output, hidden)
#         return output, hidden
#
#
#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)
# # 调用验证
# hidden_size = 25
# input_size = 20
# # pair_tensor[0]代表源语言即英文的句子，
# # pair_tensor[0][0]代表句子中的第一个词
# input = pair_tensor[0][0]
# # print('input==', input)
# # 初始化第一个隐层张量，1x1xhidden_size的0张量
# hidden = torch.zeros(1, 1, hidden_size)
#
# encoder = EncoderRNN(input_size, hidden_size)
# encoder_output, hidden = encoder(input, hidden)
# # print('encoder_output==', encoder_output)
#
#
# class DecoderRNN(nn.Module):
#
#     def __init__(self, hidden_size, output_size):
#         super(DecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         # self.embedding = nn.Embedding(output_size, hidden_size)
#         #  这里的hidden_size也是可以自己设置的
#         self.embedding = nn.Embedding(output_size, 10)
#         self.gru = nn.GRU(10, hidden_size)
#         # self.gru = nn.GRU(hidden_size, hidden_size)
#         self.out = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)
#
#
#     def forward(self, input, hidden):
#         output = self.embedding(input).view(1, 1, -1)
#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)
#         # 试试这里不进行降维可以否, 可以不降维度， 但是效果不明显
#         # output = self.softmax(self.out(output[0]))
#         output = self.softmax(self.out(output[0]))
#         return output, hidden
#
#     def initHidden(self):
#
#         return torch.zeros(1, 1, self.hidden_size, device=device)
#
# # 调用验证
# hidden_size = 25
# output_size = 10
#
# # pair_tensor[1]代表目标语言即法文的句子，pair_tensor[1][0]代表句子中的第一个词
# input = pair_tensor[1][0]
# # 初始化第一个隐层张量，1x1xhidden_size的0张量
# hidden = torch.zeros(1, 1, hidden_size)
# decoder = DecoderRNN(hidden_size, output_size)
# output, hidden = decoder(input, hidden)
# print(output)
# print(output.shape) # output是二维结构
#
# class AttnDecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
#         super(AttnDecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.dropout_p = dropout_p
#         self.max_length = max_length
#
#         self.embedding = nn.Embedding(self.output_size, self.hidden_size)
#         self.attn = nn.Linear(self.hidden_size*2, self.max_length)
#         self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         # self.attn_combine = nn.Linear(self.hidden_size * 2, 10)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#         # self.gru = nn.GRU(10, self.hidden_size)
#         self.out = nn.Linear(self.hidden_size, self.output_size)
#
#     def forward(self, input, hidden, encoder_outputs):
#         embedded = self.embedding(input).view(1, 1, -1)
#         embedded = self.dropout(embedded)
#
#         attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
#         attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
#         output = torch.cat((embedded[0], attn_applied[0]), 1)
#         output = self.attn_combine(output).unsqueeze(0)
#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)
#         output = F.log_softmax(self.out(output[0]), dim=1)
#         return output, hidden, attn_weights
#
#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)
#
# # 调用验证
# hidden_size = 25
# output_size = 10
# input = pair_tensor[1][0]
# hidden = torch.zeros(1, 1, hidden_size)
# # encoder_outputs需要是encoder中每一个时间步的输出堆叠而成
# # 它的形状应该是10x25, 我们这里直接随机初始化一个张量
# encoder_outputs  = torch.randn(10, 25)
# decoder = AttnDecoderRNN(hidden_size, output_size)
# output, hidden, attn_weights= decoder(input, hidden, encoder_outputs)
# print(output) # 这里的输出也是二维张量
#
# # 第四步: 构建模型训练函数, 并进行训练.
# teacher_forcing_ratio = 0.5
# def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
#     encoder_hidden = encoder.initHidden()
#     encoder_optimizer.zero_grad()
#     decoder_optimizer.zero_grad()
#
#     print('input_tensor==', input_tensor)
#     '''
#     input_tensor== tensor([[  2],
#         [  3],
#         [800],
#         [  4],
#         [  1]])
#     '''
#     input_length = input_tensor.size(0)
#     print('input_length==', input_length) # input_length== 5
#     target_length = target_tensor.size(0)
#     print('target_length==', target_length) # target_length== 6
#
#     encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
#
#     loss = 0
#
#     for ei in range(input_length):
#         # print('ei------', ei)
#         encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
#         # print('encoder_output-----', encoder_output) # 形状为[1, 1, hidden_size], 所以下面直接来[0, 0]进行提取
#         encoder_outputs[ei] = encoder_output[0, 0]
#     # print('encoder_outputs++++++', encoder_outputs)
#     # print('encoder_outputsshape++++++', encoder_outputs.shape) # encoder_outputsshape++++++ torch.Size([10, 256])
#
#     decoder_input = torch.tensor([[SOS_token]], device=device)
#
#     decoder_hidden = encoder_hidden
#
#     use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
#
#     if use_teacher_forcing:
#         for di in range(target_length):
#             decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
#
#             loss += criterion(decoder_output, target_tensor[di])
#             decoder_input = target_tensor[di]
#
#     else:
#         for di in range(target_length):
#             decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
#             topv, topi = decoder_output.topk(1)
#             loss += criterion(decoder_output, target_tensor[di])
#             if topi.squeeze().item() == EOS_token:
#                 break
#
#             decoder_input = topi.squeeze().detach()
#
#     loss.backward()
#
#     encoder_optimizer.step()
#     decoder_optimizer.step()
#
#     return loss.item()/target_length
#
# # 导入时间和数学工具包
# import time
# import math
#
# def timeSince(since):
#     "获得每次打印的训练耗时, since是训练开始时间"
#     # 获得当前时间
#     now = time.time()
#     # 获得时间差，就是训练耗时
#     s = now - since
#     # 将秒转化为分钟, 并取整
#     m = math.floor(s / 60)
#     # 计算剩下不够凑成1分钟的秒数
#     s -= m * 60
#     # 返回指定格式的耗时
#     return '%dm %ds' % (m, s)
#
# import matplotlib.pyplot as plt
#
# def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learing_rate=0.01):
#     start = time.time()
#     plot_losses = []
#
#     print_loss_total = 0
#     plot_loss_total = 0
#
#     encoder_optimizer = optim.SGD(encoder.parameters(), lr=learing_rate)
#     decoder_optimizer = optim.SGD(decoder.parameters(), lr=learing_rate)
#
#     criterion = nn.NLLLoss()
#
#     for iter in range(1, n_iters+1):
#         training_pair = tensorFromPair(random.choice(pairs))
#         input_tensor = training_pair[0]
#         target_tensor = training_pair[1]
#
#         loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
#
#         print_loss_total += loss
#         plot_loss_total += loss
#
#         if iter % print_every == 0:
#             print_loss_avg = print_loss_total/print_every
#             print_loss_total=0
#             print('%s (%d %d%%) %.4f' % (timeSince(start),iter, iter / n_iters * 100, print_loss_avg))
#
#
#         if iter % plot_every == 0:
#             plot_loss_avg = plot_loss_total/plot_every
#             plot_losses.append(plot_loss_avg)
#             plot_loss_total = 0
#
#     plt.figure()
#     plt.plot(plot_losses)
#     plt.savefig('./s2s_loss.png')
#
# # 设置隐层大小为256 ，也是词嵌入维度
# hidden_size = 256
# # 通过input_lang.n_words获取输入词汇总数，与hidden_size一同传入EncoderRNN类中
# # 得到编码器对象encoder1
# encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
#
# # 通过output_lang.n_words获取目标词汇总数，与hidden_size和dropout_p一同传入AttnDecoderRNN类中
# # 得到解码器对象attn_decoder1
# attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
#
# # 设置迭代步数
# n_iters = 75000
# # 设置日志打印间隔
# print_every = 5000
#
# # 调用trainIters进行模型训练，将编码器对象encoder1，码器对象attn_decoder1，迭代步数，日志打印间隔传入其中
# # trainIters(encoder1, attn_decoder1, n_iters, print_every=print_every)
#
#
# # 第五步: 构建模型评估函数, 并进行测试以及Attention效果分析.
# def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
#     with torch.no_grad():
#         input_tensor = tensorFromSentence(input_lang, sentence)
#         input_length = input_tensor.size()[0]
#         encoder_hidden = encoder.initHidden()
#
#         encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
#
#         for ei in range(input_length):
#             encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
#             encoder_outputs[ei] += encoder_output[0,0]
#
#         decoder_input = torch.tensor([[SOS_token]], device=device)
#         decoder_hidden = encoder_hidden
#
#         decoder_words = []
#
#         decoder_attentions = torch.zeros(max_length, max_length)
#
#         for di in range(max_length):
#             decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
#
#             decoder_attentions[di] = decoder_attention.data
#             topv, topi = decoder_output.data.topk(1)
#             if topi.item() == EOS_token:
#                 decoder_words.append(('<EOS>'))
#                 break
#             else:
#                 decoder_words.append(output_lang.index2word[topi.item()])
#
#             decoder_input = topi.squeeze().detach()
#
#         return decoder_words, decoder_attentions[:di+1]
#
# def evaluateRandomly(encoder, decoder, n=6):
#     for i in range(n):
#         pair = random.choice(pairs)
#         # > 代表输入
#         print('>', pair[0])
#         # = 代表正确的输出
#         print('=', pair[1])
#         # 调用evaluate进行预测
#         output_words, attentions = evaluate(encoder, decoder, pair[0])
#         # 将结果连成句子
#         output_sentence = ' '.join(output_words)
#         # < 代表模型的输出
#         print('<', output_sentence)
#         print('')
#
# evaluateRandomly(encoder1, attn_decoder1)
#
# sentence = "we re both teachers ."
# # 调用评估函数
# output_words, attentions = evaluate(
# encoder1, attn_decoder1, sentence)
# print(output_words)
# # 将attention张量转化成numpy, 使用matshow绘制
# plt.matshow(attentions.numpy())
# # 保存图像
# plt.savefig("./s2s_attn.png")


# ===== RNN模型构建人名分类器 =====
#
# # 第一步: 导入必备的工具包.
# from io import open
# import glob
# import os
# import string
# import unicodedata
# import random
# import time
# import math
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
#
# # 第二步: 对data文件中的数据进行处理，满足训练要求.
# all_letters = string.ascii_letters + " .,;'"
# n_letters = len(all_letters)
# print('n_letter:', n_letters)
#
# def unicodeToAscii(s):
#     # normalize() 第一个参数指定字符串标准化的方式。 NFC表示字符应该是整体组成(比如可能的话就使用单一编码)，而NFD表示字符应该分解为多个组合字符表示。
#     # Python同样支持扩展的标准化形式NFKC和NFKD，它们在处理某些字符的时候增加了额外的兼容特性。
#     return ''.join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn" and c in all_letters)
#
# # 调用验证
# # s = "Ślusàrski"
# # a = unicodeToAscii(s)
# # print(a)
#
# data_path = './data/names/'
#
# def readLines(filename):
#     lines = open(filename, encoding='utf-8').read().strip().split('\n')
#     return [unicodeToAscii(line) for line in lines]
#
# # 调用验证
# # filename = data_path + 'Chinese.txt'
# # filename = data_path + 'Italian.txt'
# # lines = readLines(filename)
# # print(lines)
#
# category_lines = {}
# all_categories = []
# # glob资料：https://blog.csdn.net/qq_17753903/article/details/82180227
# for filename in glob.glob(data_path + "*.txt"):
#     # os.path.basename():返回path最后的文件名。如果path以／或\结尾，那么就会返回空值,等同于os.path.split(path)的第二个元素。
#     '''
#     >>> import os
#     >>> path = '/Users/beazley/Data/data.csv'
#     >>> # Get the last component of the path
#     >>> os.path.basename(path)
#     'data.csv'
#     '''
#     # os.path.splitext：分离文件名和扩展名， 返回两个元素（文件名， 扩展名）
#     # print('os.path.basename：', os.path.basename(filename))
#     # print('os.path.splitext：', os.path.splitext(os.path.basename(filename)))
#     category = os.path.splitext(os.path.basename(filename))[0]
#     all_categories.append(category)
#     lines = readLines(filename)
#     category_lines[category] =  lines
#
# # 调用验证
# # print('category_lines', category_lines)
# n_categories = len(all_categories)
# # print('n_categories:', n_categories)
# #
# # print(category_lines['Italian'][:5])
#
# def lineToTensor(line):
#     # n_letters：57维度
#     tensor = torch.zeros(len(line), 1, n_letters)
#     for li, letter in enumerate(line):
#         tensor[li][0][all_letters.find(letter)] = 1
#
#     return tensor
#
# # 调用验证
# # line = 'Bai'
# # line_tensor = lineToTensor(line)
# #
# # print("line_tensor", line_tensor)
#
#
# # 第三步: 构建RNN模型(包括传统RNN, LSTM以及GRU).
#
# class RNN(nn.Module):
#     def __init__(self, input_size,  hidden_size, output_size, num_layers=1):
#         super(RNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers)
#         self.linear = nn.Linear(hidden_size, output_size)
#         # 在softmax层外加了log函数
#         self.softmax = nn.LogSoftmax(dim=-1)
#
#     def forward(self, input, hidden):
#         input = input.unsqueeze(0)
#         # input = input
#         rr, hn = self.rnn(input, hidden)
#         return self.softmax(self.linear(rr)), hn
#
#
#     def initHidden(self):
#         return torch.zeros(self.num_layers, 1, self.hidden_size)
#
# class LSTM(nn.Module):
#
#     def __init__(self, input_size, hidden_size, output_size, num_layers=1):
#         super(LSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
#         self.linear = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=-1)
#
#
#     def forward(self, input, hidden, c):
#         input = input.unsqueeze(0)
#         # input = input
#         rr, (hn, c) = self.lstm(input, (hidden, c))
#         return self.softmax(self.linear(rr)), hn, c
#
#
#     def initHiddenAndC(self):
#
#         c = hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
#         return hidden, c
#
#
# class GRU(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=1):
#         super(GRU, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#
#         self.gru = nn.GRU(input_size, hidden_size, num_layers)
#         self.linear = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=-1)
#
#     def forward(self, input, hidden):
#         input = input.unsqueeze(0)
#         # input = input
#         rr, hn = self.gru(input, hidden)
#         return self.softmax(self.linear(rr)), hn
#
#
#     def initHidden(self):
#         return torch.zeros(self.num_layers, 1, self.hidden_size)
#
# # 调用验证
# input_size = n_letters
# n_hidden = 128
# output_size = n_categories
#
# input = lineToTensor("B").squeeze(0)
# # input = lineToTensor("B")
# hidden = c = torch.zeros(1, 1, n_hidden)
#
# rnn = RNN(input_size, n_hidden, n_categories)
# lstm = LSTM(n_letters, n_hidden, n_categories)
# gru = GRU(n_letters, n_hidden, n_categories)
#
# rnn_output, next_hidden = rnn(input, hidden)
# print('rnn:', rnn_output)
# # print('rnn_size:', rnn_output.size()) # rnn_size: torch.Size([1, 1, 18])
# # print('squeeze:', rnn_output.squeeze(0).size()) # squeeze: torch.Size([1, 18])
# lstm_output, next_hidden, c = lstm(input, hidden, c)
# # print("lstm:", lstm_output)
# gru_output, next_hidden = gru(input, hidden)
# # print('gru:', gru_output)
#
#
# # 第四步: 构建训练函数并进行训练.
#
# def categoryFromOutput(output):
#     # pytorch.topk()用于返回Tensor中的前k个元素以及元素对应的索引值
#     top_n, top_i = output.topk(1)
#     category_i = top_i[0].item()
#     return all_categories[category_i], category_i
#
# output = gru_output
# category, category_i = categoryFromOutput(output)
# # print('category:', category)
# # print("category_i", category_i)
#
#
# def randomTrainingExample():
#     category = random.choice(all_categories)
#     line = random.choice(category_lines[category])
#     category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
#     line_tensor = lineToTensor(line)
#     return category, line, category_tensor, line_tensor
#
# # 调用验证
# # for i in range(2):
# #     category, line, category_tensor, line_tensor = randomTrainingExample()
# #     print('category=', category, '/ line =', line, '/ category_tensor =', category_tensor ,'/ line_tensor =', line_tensor,)
# #     print('category_tensor.size==:', category_tensor.size())
# #     print('line_tensor.size==:', line_tensor.size())
# #     print('line_tensor.size==:', line_tensor.size()[0])
#
# # nn.NLLLoss(): https://blog.csdn.net/jeremy_lf/article/details/102725285
# criterion = nn.NLLLoss()
#
# learning_rate = 0.005
#
# def trainRNN(category_tensor, line_tensor):
#     hidden = rnn.initHidden()
#     rnn.zero_grad()
#
#     for i in range(line_tensor.size()[0]):
#         output, hidden = rnn(line_tensor[i], hidden)
#     # 表示的意义为： 在[1, 18]一行18列中， 取出和类别一样的那个预测值和真实值进行损失值的求解过程
#     loss = criterion(output.squeeze(0), category_tensor)
#
#     loss.backward()
#
#     for p in rnn.parameters():
#         p.data.add_(-learning_rate, p.grad.data)
#
#     return output, loss.item()
#
# def trainLSTM(category_tensor, line_tensor):
#     hidden, c = lstm.initHiddenAndC()
#     lstm.zero_grad()
#
#     for i in range(line_tensor.size()[0]):
#         output, hidden, c =  lstm(line_tensor[i], hidden, c)
#
#     loss = criterion(output.squeeze(0), category_tensor)
#     loss.backward()
#
#     for p in lstm.parameters():
#         p.data.add_(-learning_rate, p.grad.data)
#
#     return output, loss.item()
#
# def trainGRU(category_tensor, line_tensor):
#     hidden = gru.initHidden()
#     gru.zero_grad()
#     for i in range(line_tensor.size()[0]):
#         output, hidden = gru(line_tensor[i], hidden)
#
#     loss = criterion(output.squeeze(0), category_tensor)
#     loss.backward()
#
#     for p in gru.parameters():
#         p.data.add_(-learning_rate, p.grad.data)
#
#     return output, loss.item()
#
# def timeSince(since):
#     now = time.time()
#     s = now - since
#     m = math.floor(s/60)
#     s -= m * 60
#     return '%dm %ds' % (m, s)
#
# # 调用验证
# # since = time.time() - 10* 60
# # period = timeSince(since)
# # print(period)
#
# n_iters = 1000
# print_every = 50
# plot_every = 10
#
# def train(train_type_fn):
#     all_losses = []
#     start = time.time()
#     current_loss = 0
#     for iter in range(1, n_iters+1):
#         category, line, category_tensor, line_tensor = randomTrainingExample()
#         output, loss = train_type_fn(category_tensor, line_tensor)
#         current_loss += loss
#         if iter % print_every  == 0:
#             guess, guess_i = categoryFromOutput(output)
#
#             correct = '✓' if guess == category else '✗ (%s)' % category
#             print(
#             '%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
#
#         if iter % plot_every == 0:
#             all_losses.append(current_loss/plot_every)
#             current_loss = 0
#     return all_losses, int(time.time() - start)
#
# # 调用训练函数
# all_losses1, period1 = train(trainRNN)
# all_losses2, period2 = train(trainRNN)
# all_losses3, period3 = train(trainRNN)
#
# # 创建画布0
# plt.figure(0)
# # 绘制损失对比曲线
# plt.plot(all_losses1, label="RNN")
# plt.plot(all_losses2, color="red", label="LSTM")
# plt.plot(all_losses3, color="orange", label="GRU")
# plt.legend(loc='upper left')
#
# plt.figure(1)
# x_data = ['RNN', 'LSTM', "GRU"]
# y_data = [period1, period2, period3]
#
# plt.bar(range(len(x_data)), y_data, tick_label=x_data)
#
# # 第五步: 构建评估函数并进行预测.
# def evaluateRNN(line_tensor):
#     hidden = rnn.initHidden()
#     for i in range(line_tensor.size()[0]):
#         output, hidden = rnn(line_tensor[i], hidden)
#         return output.squeeze(0)
#
# def evaluateLSTM(line_tensor):
#     hidden, c = lstm.initHiddenAndC()
#     for i in range(line_tensor.size()[0]):
#         output, hidden, c = lstm(line_tensor[i], hidden, c)
#
#     return output.squeeze(0)
#
# def evaluateGRU(line_tensor):
#     hidden = gru.initHidden()
#     for i in range(line_tensor.size()[0]):
#         output, hidden = gru(line_tensor[i], hidden)
#
#     return output.squeeze(0)
#
# #  调用验证
# # line = 'Bai'
# # line_tensor = lineToTensor(line)
# #
# # rnn_output = evaluateRNN(line_tensor)
# # lstm_output = evaluateLSTM(line_tensor)
# # gru_output = evaluateGRU(line_tensor)
#
# # print("rnn_output:", rnn_output)
# # print("gru_output:", lstm_output)
# # print("gru_output:", gru_output)
#
# def predict(input_line, evaluate, n_predictions=3):
#     print('\n> %s' % input_line)
#
#     with torch.no_grad():
#         output = evaluate(lineToTensor(input_line))
#         topv, topi = output.topk(n_predictions, 1, True)
#         predictions = []
#         for i in range(n_predictions):
#
#             value = topv[0][i].item()
#             category_index = topi[0][i].item()
#             print('(%.2f) %s' % (value, all_categories[category_index]))
#             predictions.append([value, all_categories[category_index]])
# #
# for evaluate_fn in [evaluateRNN, evaluateLSTM, evaluateGRU]:
#     print("-" * 18)
#     predict('Dovesky', evaluate_fn)
#     predict('Jackson', evaluate_fn)
#     predict('Satoshi', evaluate_fn)


# # ===== 注意力机制 =====
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class Attn(nn.Module):
#
#     def __init__(self, query_size, key_size, value_size1, value_size2, output_size):
#         super(Attn, self).__init__()
#         self.query_size = query_size
#         self.key_size = key_size
#         self.value_size1 = value_size1
#         self.value_size2 = value_size2
#         self.output_size = output_size
#
#         self.attn = nn.Linear(self.query_size + self.key_size, value_size1)
#         self.attn_combine = nn.Linear(self.query_size + self.value_size2, output_size)
#
#     def forward(self, Q, K, V):
#         attn_weights = F.softmax(self.attn(torch.cat((Q[0], K[0]), 1)), dim=1)
#
#         attn_applied =  torch.bmm(attn_weights.unsqueeze(0), V)
#         output = torch.cat((Q[0], attn_applied[0]), 1)
#         output = self.attn_combine(output).unsqueeze(0)
#
#         return output, attn_weights
#
# #  调用
# query_size = 32
# key_size = 32
# value_size1 = 32
# value_size2 = 64
# output_size = 64
#
# attn = Attn(query_size, key_size, value_size1, value_size2, output_size)
#
# Q = torch.randn(1, 1, 32)
# K = torch.randn(1, 1, 32)
# V = torch.randn(1, 32, 64)
# out = attn(Q, K, V)
# print(out[0])
# print(out[1])




# # ===== GRU模型 =====
# import torch
# import torch.nn as nn
#
# '''
# 第一个参数：input_size(输入张量x的维度)
# 第二个参数：hidden_size(隐藏层的维度， 隐藏层神经元的个数)
# 第三个参数：num_layers(隐藏层的层数)
# '''
# gru = nn.GRU(5, 6, 2)
# '''
# 第一个参数：sequence_length(输入序列的长度)
# 第二个参数：batch_siz(批次样本的数量)
# 第三个参数：input_size(输入张量x的维度)
# '''
# input2 = torch.randn(1, 3, 5)
# '''
# 第一个参数：num_layers * num_directions(隐藏层的层数 * 方向数)
# 第二个参数：batch_size(批次样本的数量)
# 第三个参数：hidden_size(隐藏层的维度， 隐藏层的神经元个数)
# '''
# h0 = torch.randn(2, 3, 6)
#
# output, hn = gru(input2, h0)
#
# print(output)
# print(output.shape)
# print(hn)
# print(hn.shape)

# # ===== LSTM模型 =====
# import torch
# import torch.nn as nn
#
# '''
# 第一个参数：input_size(输入张量x的维度)
# 第二个参数：hidden_size(隐藏层的维度， 隐藏层的神经元个数)
# 第三个参数：num_layer(隐藏层层数)
# '''
# lstm = nn.LSTM(5, 6, 2)
# '''
# 第一个参数：sequence_length(输入序列的长度)
# 第二个参数：batch_size(批次的样本数量)
# 第三个参数：input_size(输入张量x的维度)
# '''
# input1 = torch.randn(1, 3, 5)
# '''
# 第一个参数：num_layer * num_directions(隐藏层层数*方向数)
# 第二个参数：batch_size(批次的样本数量)
# 第三个参数：num_layer(隐藏层的维度)
# '''
# h0 = torch.randn(2, 3, 6)
# c0 = torch.randn(2, 3, 6)
# # 将input1,  h0, c0输入到lstm中， 输出结果
# output, (hn, cn) = lstm(input1, (h0, c0))
#
# print(output)
# print(output.shape)
# print(hn)
# print(hn.shape)
# print(cn)
# print(cn.shape)


# ===== RNN架构 =====
# import torch
# import torch.nn as nn
#
# '''
# 第一个参数：input_size(输入张量x的维度)
# 第二个参数：hidden_size(隐藏层的维度， 隐藏层的神经元个数)
# 第三个参数：num_layer(隐藏层的数量)
# '''
# rnn = nn.RNN(5, 6, 1)
# '''
# 第一个参数：sequence_length(输入序列的长度)
# 第二个参数：batch_size(批次的样本数量)
# 第三个参数：input_size(输入张量的维度)
# '''
# input = torch.randn(1, 3, 5)
# '''
# 第一个参数：num_layer * num_directions(层数*网络方向)
# 第二个参数：batch_size(批次的样本数)
# 第三个参数：hidden_size(隐藏层的维度， 隐藏层神经元的个数)
# '''
# h0 = torch.randn(1, 3, 6)
# # 输入input到RNN中，得到结果
# output, hn = rnn(input, h0)
#
# print(output)
# print(output.shape)
# print(hn)
# print(hn.shape)


# # ===== 新闻主题分类任务 =====
# import torch
# import torchtext
# from torchtext.datasets import text_classification
# import os
#
# from torchtext.datasets.text_classification import _csv_iterator, _create_data_from_iterator, TextClassificationDataset
# from torchtext.utils import extract_archive
# from torchtext.vocab import build_vocab_from_iterator, Vocab
#
# load_data_path = './data_new'
#
# if not os.path.isdir(load_data_path):
#     os.mkdir(load_data_path)
# # 下载不下来
# # train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](root=load_data_path)
#
# def setup_datasets(dataset_tar='./data_new/ag_news_csv.tar.gz', ngrams=1, vocab=None, include_unk=False):
#     extracted_files = extract_archive(dataset_tar)
#
#     for fname in extracted_files:
#         if fname.endswith('train.csv'):
#             train_csv_path = fname
#         if fname.endswith('test.csv'):
#             test_csv_path = fname
#
#     if vocab is None:
#         # logging.info('Building Vocab based on {}'.format(train_csv_path))
#         vocab = build_vocab_from_iterator(_csv_iterator(train_csv_path, ngrams))
#
#     else:
#         if not isinstance(vocab, Vocab):
#             raise TypeError("Passed vocabulary is not of type Vocab")
#     # logging.info('Vocab has {} entries'.format(len(vocab)))
#     # logging.info('Creating training data')
#     train_data, train_labels = _create_data_from_iterator(
#         vocab, _csv_iterator(train_csv_path, ngrams, yield_cls=True), include_unk)
#     # logging.info('Creating testing data')
#     test_data, test_labels = _create_data_from_iterator(
#         vocab, _csv_iterator(test_csv_path, ngrams, yield_cls=True), include_unk)
#     if len(train_labels ^ test_labels) > 0:
#         raise ValueError("Training and test labels don't match")
#     return (TextClassificationDataset(vocab, train_data, train_labels),
#             TextClassificationDataset(vocab, test_data, test_labels))
#
# train_dataset, test_dataset = setup_datasets()
# print("train_dataset", train_dataset)
#
#
# # 第一步: 构建带有Embedding层的文本分类模型.
# import torch.nn as nn
# import torch.nn.functional as F
#
# BATCH_SIZE = 16
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# class TextSentiment(nn.Module):
#
#     def __init__(self, vocab_size, embed_dim, num_class):
#         super.__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
#         self.fc = nn.Linear(embed_dim, num_class)
#         self.init_weights()
#
#     def init_weights(self):
#
#         initrange = 0.5
#         self.embedding.weight.data.uniform_(-initrange, initrange)
#         self.fc.weight.data.uniform_(-initrange, initrange)
#         self.fc.bias.data.zero_()
#
#     def forward(self, text):
#         embedded = self.embedding(text)
#         c = embedded.size(0) // BATCH_SIZE
#         embedded = embedded[:BATCH_SIZE*c]
#         embedded = embedded.transpose(1, 0).unsqueeze(0)
#         embedded = F.avg_pool1d(embedded, kernel_size=c)
#         return self.fc(embedded[0].transpose(1, 0))
#
# VOCAB_SIZE = len(train_dataset.get_vocab())
# EMBED_DIM=32
# NUM_CLASS = len(train_dataset.get_labels())
# model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)
#
# # 第二步: 对数据进行batch处理.
# def generate_batch(batch):
#     label = torch.tensor([entry[0] for entry in batch])
#     text = [entry[1] for entry in batch]
#     text = torch.cat(text)
#
#     return text, label
#
# batch = [(1, torch.tensor([3, 23, 2, 8])), (0, torch.tensor([3, 45, 21, 6]))]
# res = generate_batch(batch)
# print(res)
#
# # 第三步: 构建训练与验证函数.
# from torch.utils.data import DataLoader
# def train(train_data):
#     train_loss = 0
#     train_acc = 0
#
#     data = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
#
#     for i, (text, cls) in enumerate(data):
#         optimizer.zero_grad()
#         output = model(text)
#         loss = criterion(output, cls)
#         train_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#         train_acc += (output.argmax(1) == cls).sum().item()
#
#     scheduler.step()
#
#     return train_loss / len(train_data), train_acc / len(train_data)
#
# def valid(valid_data):
#     loss = 0
#     acc = 0
#     data = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=generate_batch)
#
#     for text, cls in data:
#         with torch.no_grad():
#             output = model(text)
#             loss = criterion(output, cls)
#             loss += loss.item()
#             acc += (output.argmax(1) == cls).sum().item()
#
#     return loss / len(valid_data), acc / len(valid_data)
#
#
# # 第四步: 进行模型训练和验证.
# import time
# from torch.utils.data.dataset import random_split
# N_EPOCHS = 10
# min_valid_loss = float('inf')
# criterion = torch.nn.CrossEntropyLoss().to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
#
# train_len = int(len(train_dataset) * 0.95)
#
# sub_train_, sub_valid_ = random_split(train_dataset, [train_len, len(train_dataset)-train_len])
#
# for epoch in range(N_EPOCHS):
#     start_time = time.time()
#     train_loss, train_acc = train(sub_train_)
#     valid_loss, valid_acc = valid(sub_valid_)
#
#     secs = int(time.time()-start_time)
#     mins = secs / 60
#     secs = secs % 60
#
#     print('Epoch: %d' % (epoch + 1), " | time in %d minutes, %d seconds" % (mins, secs))
#     print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
#     print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')


# 第五步: 查看embedding层嵌入的词向量.


# ===== 文本特征处理  =====

# ngram_range = 2
# def create_ngram_set(input_list):
#     return set(zip(*[input_list[i:] for i in range(ngram_range)]))
#
# input_list = [1, 3,  2, 1, 5, 3]
# res = create_ngram_set(input_list)
# print(set)
#
# from keras.preprocessing import sequence
#
# cutlen = 10
# def padding(x_train):
#     return sequence.pad_sequences(x_train, cutlen)
#
# x_train = [[1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1],[2, 32, 1, 23, 1]]
#
# res = padding(x_train)
# print(res)
#
# # 假设取两条已经存在的正样本和两条负样本
# # 将基于这四条样本产生新的同标签的四条样本
# p_sample1 = "酒店设施非常不错"
# p_sample2 = "这家价格很便宜"
# n_sample1 = "拖鞋都发霉了, 太差了"
# n_sample2 = "电视不好用, 没有看到足球"
#
# from googletrans import Translator
# translator = Translator()
#
#
# translations = translator.translate([p_sample1, p_sample2, n_sample1, n_sample2], dest='ko')
# ko_res = list(map(lambda x:x.text, translations))
# # 打印结果
# print("中间翻译结果:")
# print(ko_res)
#
# translations = translator.translate(ko_res, dest='zh-cn')
# cn_res = list(map(lambda x:x.text, translations))
# print("回译得到的增强数据:")
# print(cn_res)



# # ===== 文本数据分析 =====
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt
#
# plt.style.use('fivethirtyeight')
#
# train_data = pd.read_csv('./cn_data/train.tsv', sep='\t')
# valid_data = pd.read_csv('./cn_data/dev.tsv', sep='\t')
#
# sns.countplot('label', data=train_data)
# plt.title('train_data')
# plt.show()
#
# sns.countplot('label', data=valid_data)
# plt.title('valid_data')
# plt.show()
#
# train_data['sentence_length'] = list(map(lambda x: len(x), train_data['sentence']))
#
# sns.countplot('sentence_length', data=train_data)
# plt.xticks([])
# plt.show()
#
# sns.displot(train_data['sentence_length'])
# plt.yticks([])
# plt.show()
#
# valid_data['sentence_length'] = list(map(lambda x: len(x), valid_data['sentence']))
#
# sns.countplot('sentence_length', data=valid_data)
#
# plt.xticks([])
# plt.show()
#
# sns.displot(valid_data['sentence_length'])
# plt.yticks([])
# plt.show()
#
# sns.stripplot(y='sentence_length', x='label', data=train_data)
# plt.show()
#
# sns.stripplot(y='sentence_length', x='label', data=valid_data)
# plt.show()
#
# import jieba
# from itertools import chain
#
# train_vocab = set(chain(*map(lambda x: jieba.lcut(x), train_data['sentence'])))
# print("训练集共包含不同词汇总数为：", len(train_vocab))
#
# valid_vocab = set(chain(*map(lambda x: jieba.lcut(x), valid_data['sentence'])))
# print("训练集共包含不同词汇总数为：", len(valid_vocab))
#
# import jieba.posseg as  pseg
#
# def get_a_list(text):
#     r = []
#     for g in pseg.lcut(text):
#         if g.flag == 'a':
#             r.append(g.word)
#
#     return r
#
# from wordcloud import WordCloud
#
# def get_word_cloud(keywords_list):
#     wordcloud = WordCloud(font_path='./SimHei.ttf', max_words=100,  background_color='white')
#     keywords_string = ' '.join(keywords_list)
#     wordcloud.generate(keywords_string)
#
#     plt.figure()
#     plt.imshow(wordcloud, interpolation='biliner')
#     plt.axis('off')
#     plt.show()
#
# p_train_data = train_data[train_data['label']==1]['sentence']
#
# train_p_a_vocab = chain(*map(lambda x: get_a_list(x), p_train_data))
#
# n_train_data = train_data[train_data['label']==0]['sentence']
# train_n_a_vocab = chain(*map(lambda x:get_a_list(x), n_train_data))
#
# get_word_cloud(train_p_a_vocab)
# get_word_cloud(train_n_a_vocab)
#
# p_valid_data = valid_data[valid_data['label']==1]['sentence']
# valid_p_a_vocab = chain(*map(lambda x: get_a_list(x), p_valid_data))
#
# n_valid_data = valid_data[valid_data['label']==0]['sentence']
# valid_n_a_vocab = chain(*map(lambda x:get_a_list(x), n_valid_data))
#
# get_word_cloud(p_valid_data)
# get_word_cloud(n_valid_data)


# =====分词=====
# import jieba
# content = "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作"

# 精确模式分词
# print(jieba.cut(content, cut_all=False))
# print(jieba.lcut(content, cut_all=False))

# # 全模式分词
# print(jieba.cut(content, cut_all=True))
# print(jieba.lcut(content, cut_all=True))
#
# # 搜索引擎模式分词
# print(jieba.cut_for_search(content))
# print(jieba.lcut_for_search(content))
#
# # 中文繁体
# content = "煩惱即是菩提，我暫且不提"
# print(jieba.lcut(content))

# 使用自定义词典
# print(jieba.lcut("八一双鹿更名为八一南昌篮球队！"))
# jieba.load_userdict("./userdict.txt")
# print(jieba.lcut("八一双鹿更名为八一南昌篮球队！"))

# 注意环境没有配好
# import hanlp
# tokenizer = hanlp.load('CTB6_CONVSEG')
# print(tokenizer("工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作"))
#
# tokenizer = hanlp.utils.rules.tokenize_english
# print(tokenizer('Mr. Hankcs bought hankcs.com for 1.5 thousand dollars.'))
#
# import hanlp
# recognizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)
# recognizer(list('上海华安工业（集团）公司董事长谭旭光和秘书张晚霞来到美国纽约现代艺术博物馆参观。'))
#
# recognizer = hanlp.load(hanlp.pretrained.ner.CONLL03_NER_BERT_BASE_UNCASED_EN)
# recognizer(["President", "Obama", "is", "speaking", "at", "the", "White", "House"])

# import jieba.posseg as  pseg
#
# print(pseg.lcut("我爱北京天安门"))

# import hanlp
# tagger = hanlp.load(hanlp.pretrained.pos.CTB5_POS_RNN_FASTTEXT_ZH)
# tagger(['我', '的', '希望', '是', '希望', '和平'])

# import hanlp
# tagger = hanlp.load(hanlp.pretrained.pos.PTB_POS_RNN_FASTTEXT_EN)
# tagger(['I', 'banked', '2', 'dollars', 'in', 'a', 'bank', '.'])
#

# import fasttext
# model = fasttext.train_unsupervised('data/fil9')
# model.get_word_vector("the")







