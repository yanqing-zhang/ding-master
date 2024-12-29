'''
@Project ：transformer-master 
@File    ：embeddings.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/14 17:27 
'''
import torch
# 预定义的网络层torch.nn,工具开发者已经帮我们开发好了一些常用层，
# 比如， 卷积层， lstm层， embedding层等， 不需要我们再重新造轮子了
import torch.nn as nn
import math
# torch 中变量封装函数Variable
from torch.autograd import Variable

class Embeddings(nn.Module):
    """
    定义Embedings类来实现文本嵌入层，这里s说明代表两个一模一样的嵌入层，他们共享参数。
    该类继承nn.Module,这样就有标准层的一些功能，这里我们也可以理解为一种模式，我们自己实现的所有层
    """
    def __init__(self, d_model, vocab):
        """类的初始化函数， 有两个参数， d_model:指词嵌入的维度，vocab:指词表的大小。"""
        # 接着就是使用super的方式指明继承nn.Module的初始化函数， 我们自己实现的所有层都会这样去实现
        super(Embeddings, self).__init__()
        # 之后就是调用nn的预定义层Embedding，获得一个词嵌入对象self.lut
        self.lut = nn.Embedding(vocab, d_model)
        # 最后就是将d_model传入类中
        self.d_model = d_model

    def forward(self, x):
        """可以将其理解为该层的前向传播逻辑，所有层中都会有此函数
        当传给该类的实例化对象参数时，自动调用该类函数
        参数x: 因为Embedding层是首层， 所以代表输入给模型的文本通过词汇映射后的张量"""

        # 将x传给self.lut 并与根号下self.d_model相乘作为结果返回
        return self.lut(x) * math.sqrt(self.d_model)


def embeddings_testdata():
    """
    词嵌入测试数据制造
    输入参数:
    d_model:512(词嵌入维度)
    vocab:1000(词表大小)
    x:2 × 4的张量(2行4列)
    :return:embr:返回一个embedding后的值(2 × 4 × 512)
    """
    # 词嵌入维度是512维
    d_model = 512
    # 词表大小是1000
    vocab = 1000
    # 输入x是一个使用Variable封装的长整型张量， 形状是2 × 4
    x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))

    emb = Embeddings(d_model, vocab)
    embr = emb(x)
    return embr

if __name__ == '__main__':
    # 词嵌入维度是512维
    d_model = 512
    # 词表大小是1000
    vocab = 1000
    # 输入x是一个使用Variable封装的长整型张量， 形状是2 × 4
    x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))

    emb = Embeddings(d_model, vocab)
    embr = emb(x)
    print(f"embr:{embr}")
    print(f"shape:{embr.shape}")