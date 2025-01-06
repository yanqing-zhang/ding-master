'''
@Project ：langchain-master 
@File    ：utils.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2025/1/6 14:41 
'''
import os

def set_proxy():
    os.environ["http_proxy"] = "http://127.0.0.1:10792"
    os.environ["https_proxy"] = "http://127.0.0.1:10792"