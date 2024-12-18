'''
@Project ：transformer-master 
@File    ：sublayer_connection.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/18 13:10 
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from layer_norm import LayerNorm
from multiheaded_attention import MultiHeadedAttention
from positional_encoding import positional_encoding_testdata
class SublayerConnection(nn.Module):
    """使用SublayerConnection来实现子层连接结构的类"""
    def __init__(self, size, dropout=0.1):
        """
        它输入参数有两个, size以及dropout， size一般是都是词嵌入维度的大小，
           dropout本身是对模型结构中的节点数进行随机抑制的比率，
           又因为节点被抑制等效就是该节点的输出都是0，因此也可以把dropout看作是对输出矩阵的随机置0的比率.
        :param size:
        :param dropout:
        """
        super(SublayerConnection, self).__init__()
        # 实例化了规范化对象self.norm
        self.norm = LayerNorm(size)
        # 又使用nn中预定义的droupout实例化一个self.dropout对象.
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        """
        前向逻辑函数中, 接收上一个层或者子层的输入作为第一个参数，
           将该子层连接中的子层函数作为第二个参数
        :param x:
        :param sublayer:
        :return:
        """
        # 我们首先对输出进行规范化，然后将结果传给子层处理，之后再对子层进行dropout操作，
        # 随机停止一些网络中神经元的作用，来防止过拟合. 最后还有一个add操作，
        # 因为存在跳跃连接，所以是将输入x与dropout后的子层输出结果相加作为最终的子层连接输出.
        return x + self.dropout(sublayer(self.norm(x)))

def sublayer_connection_testdata():
    # 调用验证
    size = 512
    dropout = 0.2
    head = 8
    d_model = 512
    # 令x为位置编码器的输出
    x = positional_encoding_testdata()
    mask = Variable(torch.zeros(8, 4, 4))

    # 假设子层中装的是多头注意力层, 实例化这个类
    self_attn = MultiHeadedAttention(head, d_model)

    # 使用lambda获得一个函数类型的子层
    sublayer = lambda x: self_attn(x, x, x, mask)
    sc = SublayerConnection(size, dropout)
    sc_result = sc(x, sublayer)
    return sc_result

if __name__ == '__main__':
    # 调用验证
    size = 512
    dropout = 0.2
    head = 8
    d_model = 512
    # 令x为位置编码器的输出
    x = positional_encoding_testdata()
    mask = Variable(torch.zeros(8, 4, 4))

    # 假设子层中装的是多头注意力层, 实例化这个类
    self_attn = MultiHeadedAttention(head, d_model)

    # 使用lambda获得一个函数类型的子层
    sublayer = lambda x: self_attn(x, x, x, mask)
    sc = SublayerConnection(size, dropout)
    sc_result = sc(x, sublayer)
    print(f"sc_result:{sc_result}")
    print(f"shape of sc_result:{sc_result.shape}")
