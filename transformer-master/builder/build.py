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
from torch import nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WikiText2
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

# 参数设置
BATCH_SIZE = 32
EMBEDDING_DIM = 512
FFN_DIM = 2048
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.1
LEARNING_RATE = 5e-4
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取分词器
tokenizer = get_tokenizer('basic_english')

# 定义处理函数
def process_text(text):
    return tokenizer(text)

# 将文本转换为数值形式
def numericalize(data, vocab):
    return [vocab(process_text(text)) for text in data]

def get_datas():
    # 加载数据集
    train_dataset, val_dataset, test_dataset = WikiText2()

    # 构建词汇表
    vocab = build_vocab_from_iterator(map(process_text, train_dataset), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    vocab.set_default_index(vocab['<unk>'])

    train_data = numericalize(train_dataset, vocab)
    val_data = numericalize(val_dataset, vocab)
    test_data = numericalize(test_dataset, vocab)
    return train_data, val_data, test_data, vocab

# 创建DataLoader
def collate_batch(batch):
    label = torch.tensor([item[1:] for item in batch])
    text = torch.tensor([item[:-1] for item in batch])
    return text, label

def get_data_loader():
    train_data, val_data, test_data, vocab = get_datas()
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    return train_loader, val_loader


if __name__ == '__main__':
    train_loader, val_loader = get_data_loader()
    for batch_idx, (text, label) in enumerate(train_loader):
        print(f'Batch index: {batch_idx}')
        print(f'Text batch shape: {text.shape}')
        print(f'Label batch shape: {label.shape}')
        # 如果你想要查看文本和标签的具体内容，可以这样做：
        # 注意：由于数据可能很大，这里只打印每个批次的前10个样本
        print(f'Text batch (first 10 samples): {text[:10]}')
        print(f'Label batch (first 10 samples): {label[:10]}')
        print('---')