'''
@Project ：transformer-master 
@File    ：build.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/19 14:29 
'''
import torch
from torchtext.datasets import WikiText2
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 定义分词函数
def tokenize(text):
    return text.split()

# 加载数据集
train_txt, val_txt, test_txt = WikiText2()

# 将数据集转换为映射风格
train_txt = to_map_style_dataset(train_txt)
val_txt = to_map_style_dataset(val_txt)
test_txt = to_map_style_dataset(test_txt)

# 定义一个函数来迭代数据集并生成词汇
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenize(text)

# 构建词汇表
vocab = build_vocab_from_iterator(yield_tokens(train_txt), specials=['<sos>', '<eos>'])

# 定义一个函数来将文本转换为张量
def text_pipeline(text):
    tokens = tokenize(text)
    return torch.tensor([vocab[token] for token in tokens], dtype=torch.long)

# 你现在可以使用 text_pipeline 函数来处理文本数据



if __name__ == '__main__':
    text_pipeline()