'''
@Project ：transformer-master 
@File    ：variable_testcase.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2025/1/6 17:40 
'''
import torch
from torch.autograd import Variable

def variable_test():
    t = torch.arange(0, 100, 2)
    x = torch.arange(1, 101, 2)
    x = x + Variable(t)
    print(f"x:{x}")
    print(f"shape of x:{x.shape}")

if __name__ == '__main__':
    variable_test()