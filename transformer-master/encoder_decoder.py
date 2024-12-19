'''
@Project ：transformer-master 
@File    ：encoder_decoder.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/19 9:36 
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from encoder import encoder_testdata
from decoder import decoder_testdata
from generate import generate_testdata
class EncoderDecoder(nn.Module):
    """使用EncoderDecoder类来实现编码器-解码器结构"""
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        """初始化函数中有5个参数, 分别是编码器对象, 解码器对象,
           源数据嵌入函数, 目标数据嵌入函数,  以及输出部分的类别生成器对象
        """
        super(EncoderDecoder, self).__init__()
        # 将参数传入到类中
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        """在forward函数中，有四个参数, source代表源数据, target代表目标数据,
           source_mask和target_mask代表对应的掩码张量
        """

        # 在函数中, 将source, source_mask传入编码函数, 得到结果后,
        # 与source_mask，target，和target_mask一同传给解码函数.
        return self.decode(self.encode(source, source_mask), source_mask,
                            target, target_mask)

    def encode(self, source, source_mask):
        """编码函数, 以source和source_mask为参数"""
        # 使用src_embed对source做处理, 然后和source_mask一起传给self.encoder
        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        """解码函数, 以memory即编码器的输出, source_mask, target, target_mask为参数"""
        # 使用tgt_embed对target做处理, 然后和source_mask, target_mask, memory一起传给self.decoder
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)

def encoder_decoder_testdata():
    # 调用验证
    vocab_size = 1000
    d_model = 512

    _, encoder = encoder_testdata()
    _, decoder = decoder_testdata()
    source_embed = nn.Embedding(vocab_size, d_model)
    target_embed = nn.Embedding(vocab_size, d_model)
    _, generator = generate_testdata()
    # 假设源数据与目标数据相同, 实际中并不相同
    source = target = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))

    # 假设src_mask与tgt_mask相同，实际中并不相同
    source_mask = target_mask = Variable(torch.zeros(8, 4, 4))

    ed = EncoderDecoder(encoder, decoder, source_embed, target_embed, generator)
    return ed

if __name__ == '__main__':
    # 调用验证
    vocab_size = 1000
    d_model = 512

    _, encoder = encoder_testdata()
    _, decoder = decoder_testdata()
    source_embed = nn.Embedding(vocab_size, d_model)
    target_embed = nn.Embedding(vocab_size, d_model)
    _, generator = generate_testdata()
    # 假设源数据与目标数据相同, 实际中并不相同
    source = target = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))

    # 假设src_mask与tgt_mask相同，实际中并不相同
    source_mask = target_mask = Variable(torch.zeros(8, 4, 4))

    ed = EncoderDecoder(encoder, decoder, source_embed, target_embed, generator)
    print(f"type of encoder:{type(ed.encoder)}")
    ed_result = ed(source, target, source_mask, target_mask)
    print(f"ed_result:{ed_result}")
    print(f"shape of ed_result:{ed_result.shape}")