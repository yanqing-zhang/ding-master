'''
@Project ：transformer-master 
@File    ：unsqueeze_testcase.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2025/1/6 16:45 
'''
import torch


def unsqueeze_test():
    """
    一维：tensor([   0,    1,    2,  ..., 4997, 4998, 4999])
    --------------------------------------------------
    二维：tensor([[   0],
        [   1],
        [   2],
        ...,
        [4997],
        [4998],
        [4999]])
    :return:
    """
    max_len = 5000
    position_1 = torch.arange(0, max_len)
    print(f"position_1:{position_1}")
    position_2 = position_1.unsqueeze(1)
    print(f"position_2:{position_2}")


if __name__ == '__main__':
    unsqueeze_test()