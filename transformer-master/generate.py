'''
@Project ：transformer-master 
@File    ：generate.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/18 15:43 
'''
import torch.nn as nn
# nn.functional工具包装载了网络层中那些只进行计算, 而没有参数的层
import torch.nn.functional as F
from decoder import decoder_testdata
class Generator(nn.Module):
    """将线性层和softmax计算层一起实现, 因为二者的共同目标是生成最后的结构
        因此把类的名字叫做Generator, 生成器类
    """
    def __init__(self, d_model, vocab_size):
        """初始化函数的输入参数有两个, d_model代表词嵌入维度, vocab_size代表词表大小."""
        super(Generator, self).__init__()

        # 首先就是使用nn中的预定义线性层进行实例化, 得到一个对象self.project等待使用,
        # 这个线性层的参数有两个, 就是初始化函数传进来的两个参数: d_model, vocab_size
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """前向逻辑函数中输入是上一层的输出张量x"""

        # 在函数中, 首先使用上一步得到的self.project对x进行线性变化,
        # 然后使用F中已经实现的log_softmax进行的softmax处理.
        # 在这里之所以使用log_softmax是因为和我们这个pytorch版本的损失函数实现有关, 在其他版本中将修复.
        # log_softmax就是对softmax的结果又取了对数, 因为对数函数是单调递增函数,
        # 因此对最终我们取最大的概率值没有影响. 最后返回结果即可.
        return F.log_softmax(self.project(x), dim=-1)

def generate_testdata():
    # 调用验证
    # 词嵌入维度是512维
    d_model = 512

    # 词表大小是1000
    vocab_size = 1000

    # 输入x是上一层网络的输出, 我们使用来自解码器层的输出
    de_result,_ = decoder_testdata()
    x = de_result

    gen = Generator(d_model, vocab_size)
    gen_result = gen(x)
    return gen_result, gen

if __name__ == '__main__':
    # 调用验证
    # 词嵌入维度是512维
    d_model = 512

    # 词表大小是1000
    vocab_size = 1000

    # 输入x是上一层网络的输出, 我们使用来自解码器层的输出
    de_result,_ = decoder_testdata()
    x = de_result

    gen = Generator(d_model, vocab_size)
    gen_result = gen(x)
    print(f"gen_result:{gen_result}")
    print(f"shape of gen_result:{gen_result.shape}")