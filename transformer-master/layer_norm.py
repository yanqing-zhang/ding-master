'''return
@Project ：transformer-master 
@File    ：layer_norm.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/18 12:21 
'''
import torch
import torch.nn as nn
from position_wise_feed_forward import positionwise_feed_forward_testdata
class LayerNorm(nn.Module):
    """通过LayerNorm实现规范化层的类"""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 根据features的形状初始化两个参数张量a2，和b2，第一个初始化为1张量，
        # 也就是里面的元素都是1，第二个初始化为0张量，也就是里面的元素都是0，这两个张量就是规范化层的参数，
        # 因为直接对上一层得到的结果做规范化公式计算，将改变结果的正常表征，因此就需要有参数作为调节因子，
        # 使其即能满足规范化要求，又能不改变针对目标的表征.最后使用nn.parameter封装，代表他们是模型的参数。
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

        # 把eps传到类中
        self.eps = eps

    def forward(self, x):
        # 在函数中，首先对输入变量x求其最后一个维度的均值，并保持输出维度与输入维度一致.
        # 接着再求最后一个维度的标准差，然后就是根据规范化公式，用x减去均值除以标准差获得规范化的结果，
        # 最后对结果乘以我们的缩放参数，即a2，*号代表同型点乘，即对应位置进行乘法操作，加上位移参数b2.返回即可.
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std +self.eps) + self.b2

def layer_norm_testdata():
    features = d_model = 512
    eps = 1e-6
    # 输入x来自前馈全连接层的输出
    ff_result = positionwise_feed_forward_testdata()
    x = ff_result
    ln = LayerNorm(features, eps)
    ln_result = ln(x)
    return ln_result

if __name__ == '__main__':
    features = d_model = 512
    eps = 1e-6
    # 输入x来自前馈全连接层的输出
    ff_result = positionwise_feed_forward_testdata()
    x = ff_result
    ln = LayerNorm(features, eps)
    ln_result = ln(x)
    print(f"ln_result:{ln_result}")
    print(f"shape of ln_result:{ln_result.shape}")


