'''
@Project ：transformer-master 
@File    ：multiheaded_attention.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/16 18:15 
'''
import torch
import torch.nn as nn
from utils import clones
from attention_utils import attention
from positional_encoding import PositionalEncoding
from torch.autograd import Variable
from embeddings import Embeddings
from positional_encoding import positional_encoding_testdata
class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # 在函数中，首先使用了一个测试中常用的assert语句，判断h是否能被d_model整除，
        # 这是因为我们之后要给每个头分配等量的词特征，也就是embedding_dim/head个
        assert embedding_dim%head== 0

        # 得到每个头获得的分割词向量维度d_k
        self.d_k = embedding_dim // head

        # 传入头数h
        self.head = head
        # 然后获得线性层对象，通过nn的Linear实例化，它的内部变换矩阵是embedding_dim × embedd
        # 为什么是四个呢，这是因为在多头注意力中，Q,K,V各需要一个，最后拼接的矩阵还需要一个，因
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        # self.attn为None，它代表最后得到的注意力张量，现在还没有结果所以为None
        self.attn = None
        # 最后就理一个self.dropout对象，它通过nn中的Dropout实例化而来，置0比率为传进来的参数
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value,  mask=None):

        if mask is not None:
            mask = mask.unsqueeze(0)
        batch_size = query.size(0)
        # view中的四个参数的意义
        # batch_size: 批次的样本数量
        # -1这个位置应该是： 每个句子的长度
        # self.head*self.d_k应该是embedding的维度， 这里把词嵌入的维度分到了每个头中， 即每个头中分到了词的部分维度的特征
        # query, key, value形状torch.Size([2, 8, 4, 64])
        query, key, value = [model(x).view(batch_size, -1,  self.head, self.d_k).transpose(1, 2) for model, x in zip(self.linears, (query, key, value))]
        # 所以mask的形状 torch.Size([1, 8, 4, 4])  这里的所有参数都是4维度的   进过dropout的也是4维度的
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # contiguous解释:https://zhuanlan.zhihu.com/p/64551412
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head*self.d_k)

        return self.linears[-1](x)

def multiheaded_attention_testdata():
    """
    多头注意力机制测试数据制造
    :return:
    """
    pe_data = positional_encoding_testdata()
    # # 头数head
    head = 8
    # # 词嵌入维度embedding_dim
    embedding_dim = 512
    # # 置零比率dropout
    dropout = 0.2
    # # 假设输入的Q，K，V仍然相等
    query = value = key = pe_data
    # # 输入的掩码张量mask
    mask = Variable(torch.zeros(8, 4, 4))
    mha = MultiHeadedAttention(head, embedding_dim, dropout)
    mha_result = mha(query, key, value, mask)
    return mha_result

if __name__ == '__main__':
    d_model = 512

    # 词表大小是1000
    vocab = 1000
    # 输入x是一个使用Variable封装的长整型张量, 形状是2 x 4 注意： 这里必须保证两句话的长度一致！！！
    x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
    emb = Embeddings(d_model, vocab)
    embr = emb(x)
    print("------------------------------------")
    dropout = 0.1
    # # 句子最大长度
    max_len = 60
    x = embr
    pe = PositionalEncoding(d_model, dropout, max_len)
    pe_result = pe(x)
    print("------------------------")
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
    print(f"mha_result:{mha_result}")
    print(f"shape of mha_result:{mha_result.shape}")











