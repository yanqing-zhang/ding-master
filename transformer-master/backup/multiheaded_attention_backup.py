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
class MultiHeadedAttention(nn.Module):

    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        # 在函数中，首先使用了一个测试中常用的assert语句，判断h是否能被d_model整除，
        # 这是因为我们之后要给每个头分配等量的词特征，也就是embedding_dim/head个
        assert embedding_dim % head == 0

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

    def forward(self, query, key, value, mask=None):
        """前向逻辑函数，它的输入参数有四个，前三个就是注意力机制需要的Q,K,V,
        最后一个是注意力机制中可能需要的mask掩码张量，默认是None"""

        # 如果存在掩码张量mask
        if mask is not None:
            # 使用unsqueeze拓展维度，代表多头中的第n头
            mask = mask.unsqueeze(1)
        # 接着，我们获得一个batch_size的变量，他是query尺寸的第1个数字，代表有多少条样本
        batch_size = query.size(0)
        # 之后就进入多头处理环节
        # 首先利用zip将输入QKV每三个线性层组到一起，然后使用for循环，将输入QKV分别传到线性层中，
        # 做完线性变换后，开始为每个头分割输入，这里使用view方法对线性变换的结果进行维度重塑，多
        # 这样就意味着每个头可以获得一部分词特征组成的句子，其中的-1代表自适应维度，
        # 计算机会根据这种变换自动计算这里的值，然后对第二维和第三维进行转置操作，
        # 为了让代表句子长度维度和词向量维度能够相邻，这样注意力机制才能找到词义与句子位置的关系
        # 从attention函数中可以看到，利用的是原始输入的倒数第一和第二维，这样我们就得到了每个头
        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
                             for model, x in zip(self.linears, (query, key, value))]
        # 得到每个头的输入后，接下来就是将他们传入到attention中
        # 这里直接通过我们之前实现的attention函数，同时也将mask和dropout传入其中。
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 通过多头注意力计算后，我们就得到了每个头计算结果组成的4维张量，我们需要将其转换为输入的
        # 因些这里开始进行第一步处理环节的逆操作，先对第二和第三维进行转置，然后使用contiguous
        # 这个方法的作用就是能够让转置后的张量应用view方法，否则将无法直接使用
        # 所以，下一步就是使用view重塑形状，变成输入形状相同
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 最后使用线性层列表中的最后一个线性层对输入进行线性变换得到最终的多头注意力结构的输出
        return self.linears[-1](x)

if __name__ == '__main__':
    d_model = 512

    # 词表大小是1000
    vocab = 1000
    # 输入x是一个使用Variable封装的长整型张量, 形状是2 x 4 注意： 这里必须保证两句话的长度一致！！！
    x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
    # x = torch.LongTensor([[100,2,421,508],[491,998,1,221]])
    # print('x===', x)
    emb = Embeddings(d_model, vocab)
    embr = emb(x)
    print("------------------------------------")
    dropout = 0.1
    #
    # # 句子最大长度
    max_len = 60
    x = embr
    #
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











