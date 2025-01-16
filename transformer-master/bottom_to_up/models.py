'''
@Project ：transformer-master 
@File    ：models.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2025/1/13 15:27 
'''
import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, enc_vocab_size, dec_vocab_size, pad_idx, d_model, num_layes, heads, d_ff, dropout=0.1,
                 max_seq_len=512):
        super().__init__()
        self.encoder = Encoder(enc_vocab_size, pad_idx, d_model, num_layes, heads, d_ff, dropout, max_seq_len)
        self.decoder = Decoder(dec_vocab_size, pad_idx, d_model, num_layes, heads, d_ff, dropout, max_seq_len)
        self.linear = nn.Linear(d_model, dec_vocab_size)
        self.pad_idx = pad_idx

    def generate_mask(self, query, key, is_triu_mask=False):
        '''
            batch,seq_len
        '''
        device = query.device
        batch, seq_q = query.shape
        _, seq_k = key.shape
        # batch,head,seq_q,seq_k
        mask = (key == self.pad_idx).unsqueeze(1).unsqueeze(2)
        mask = mask.expand(batch, 1, seq_q, seq_k).to(device)
        if is_triu_mask:
            dst_triu_mask = torch.triu(torch.ones(seq_q, seq_k, dtype=torch.bool), diagonal=1)
            dst_triu_mask = dst_triu_mask.unsqueeze(0).unsqueeze(1).expand(batch, 1, seq_q, seq_k).to(device)
            return mask | dst_triu_mask
        return mask

    def forward(self, src, dst):
        src_mask = self.generate_mask(src, src)
        encoder_out = self.encoder(src, src_mask)
        dst_mask = self.generate_mask(dst, dst, True)
        src_dst_mask = self.generate_mask(dst, src)
        decoder_out = self.decoder(dst, encoder_out, dst_mask, src_dst_mask)
        out = self.linear(decoder_out)
        return out


if __name__ == "__main__":
    # PositionEncoding(512,100)
    att = Transformer(100, 200, 0, 512, 6, 8, 1024, 0.1)
    x = torch.randint(0, 100, (4, 64))
    y = torch.randint(0, 200, (4, 64))
    out = att(x, y)
    print(out.shape)