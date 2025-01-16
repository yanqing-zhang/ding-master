'''
@Project ：transformer-master 
@File    ：utils.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/16 18:05 
'''
import copy
import torch.nn as nn
# 首先需要定义克隆函数，因为在多头注意力机制的实现中，用到多个结构相同的线性层
# 我们将使用clone函数将他们一同初始化在一个网络层列表对象中，之后的结构中也会用到该函数
def clones(module, N):
    """用于生成相同网络层的克隆函数，它的参数module表示要克隆的目标网络层，N代表需要克隆的数量"""
    # 在函数中，我们通过for循环对module进行N次深度拷贝，使其每个module成为独立的层
    # 然后将其放在nn.ModuleList类型的列表中存放
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])