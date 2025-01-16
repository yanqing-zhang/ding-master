'''
@Project ：transformer-master 
@File    ：build.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/19 14:29 
'''
import torch
import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_data():
    tokenizer = get_tokenizer("basic_english")
    train_data = WikiText2("train")
    train_iter = iter(train_data)

    for text in train_iter:
        print(f"text:{text}")
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter))


if __name__ == '__main__':
    get_data()