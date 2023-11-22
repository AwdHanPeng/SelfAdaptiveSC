import math

import torch
from torch import nn

from .utils import init_embedding_weights


class SrcEmbedding(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.args = args
        if self.args.old_PE:
            self.p = nn.Embedding(self.args.max_source_len, self.args.embedding_size)
            init_embedding_weights(self.p)
        self.embedding = nn.Embedding(len(vocab), args.embedding_size, padding_idx=vocab.pad_index)
        init_embedding_weights(self.embedding)

    def forward(self, seq):
        c = self.embedding(seq)
        if self.args.embedding_mul:
            c *= math.sqrt(self.args.embedding_size)
        if self.args.old_PE:
            seq_len = seq.size(1)
            pos = torch.arange(start=0, end=seq_len).to(seq.device)
            c = c + self.p(pos)
        c = self.dropout(c)
        return c


class TgtEmbedding(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.p = nn.Embedding(args.max_target_len + 2, args.embedding_size)
        init_embedding_weights(self.p)
        self.embedding = nn.Embedding(len(vocab), args.embedding_size, padding_idx=vocab.pad_index)
        init_embedding_weights(self.embedding)
        self.dropout = nn.Dropout(args.dropout)
        self.embedding_size = args.embedding_size
        self.args = args

    def forward(self, seq, step=None):
        c = self.embedding(seq)  # bs,l,hidden
        if self.args.embedding_mul:
            c *= math.sqrt(self.embedding_size)
        if step is None:
            seq_len = seq.size(1)
            pos = torch.arange(start=0, end=seq_len)
        else:
            # means step_wise_decode
            pos = torch.LongTensor([step])  # 1
        pos = pos.to(seq.device)
        pos_emb = self.p(pos).unsqueeze(0)  # [1, 1/seq_len, dim]
        c = c + pos_emb
        c = self.dropout(c)
        return c

# 目前右边这些是完整看完，并没有错误的代码
