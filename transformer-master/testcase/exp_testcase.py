'''
@Project ：transformer-master 
@File    ：exp_testcase.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2025/1/6 16:53 
'''
import torch
import math

def exp_test():
    d_model = 5
    a = torch.arange(0, d_model, 2) # 生成一个一维数据，从0到5，步长为2
    print(f"a:{a}")
    print(f"shape of a:{a.shape}")

    print("---------------------------------------")
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    print(f"div_term:{div_term}")
    print(f"shape of div_term:{div_term.shape}")


def multi_test():
    """
    position：tensor([[   0],
        [   1],
        [   2],
        ...,
        [4997],
        [4998],
        [4999]])
    div_term：tensor([1.0000e+00, 2.5119e-02, 6.3096e-04])
    a=position * div_term：tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00],
        [1.0000e+00, 2.5119e-02, 6.3096e-04],
        [2.0000e+00, 5.0238e-02, 1.2619e-03],
        ...,
        [4.9970e+03, 1.2552e+02, 3.1529e+00],
        [4.9980e+03, 1.2554e+02, 3.1535e+00],
        [4.9990e+03, 1.2557e+02, 3.1542e+00]])
    :return:
    """
    d_model = 5
    max_len = 5000
    position = torch.arange(0, max_len).unsqueeze(1)
    print(f"position:{position}")
    print(f"shape of position:{position.shape}") # 5000 × 1
    print("--1--------------------------------")
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    print(f"div_term:{div_term}")
    print(f"shape of div_term:{div_term.shape}") # 1 × 3
    print("--2--------------------------------")
    a = position * div_term
    print(f"a:{a}")
    print(f"shape of a:{a.shape}") # 操作符* 具有广播扩散机制， [5000 × 1] * [3] = [5000 × 3]

if __name__ == '__main__':
    multi_test()