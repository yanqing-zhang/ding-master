'''
@Project ：transformer-master 
@File    ：mask_utils.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/15 11:39 
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
def subsequent_mask(size):
    """生成向后遮掩的掩码张量，参数size是掩码张量最后两个维度的大小，它的最后两维形成一个方阵"""
    # 在函数中，首先定义掩码张量的形状
    attn_shape = (1, size, size)
    # 然后使用np.ones方法向这个形状中添加1元素，形成上三角阵，最后为了节约空间
    # 再使其中的数据类型变为无符号位整型unit8
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 最后将numpy类型转化为torch中的tensor，内部做一个 1 - 操作
    # 在这个其实是做了一个三角阵的反转，subsequent_mask中的每个元素都会被1减
    # 如果是0，subsequent_mask中的该位置由0变成1
    # 如果是1，subsequent_mask中的该位置由1变成0
    return torch.from_numpy(1 - subsequent_mask)

if __name__ == '__main__':
    size = 5
    sm = subsequent_mask(size)
    print(f"sm:{sm}")
    print(f"shape of sm:{sm.shape}")
    plt.figure(figsize=(5, 5))
    plt.imshow(subsequent_mask(20)[0])
    plt.show()

