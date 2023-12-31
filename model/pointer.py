"""adapted from https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/global_attention.py
"""

import torch
from torch import nn
import torch.nn.functional as F


class CopyAttention(nn.Module):  # CHECKED
    """ Generator module that additionally considers copying words directly from the source.
    The main idea is that we have an extended `dynamic dictionary`. It contains `|tgt_dict|`
    words plus an arbitrary number of additional words introduced by the source sentence.
    For each source sentence we have a `src_map` that maps each source word to an index
    in `tgt_dict` if it has known, or else to an extra word.
    The copy generator is an extended version of the standard generator that computes
    three values.
    - :math:`p_{softmax}` the standard softmax over `tgt_dict`
    - :math:`p(z)` the probability of copying a word from the source
    - :math:`p_{copy}` the probability of copying a particular word.
      taken from the attention distribution directly.

    The model returns a distribution over the extended dictionary, computed as:
        `p(w) = p(z=1) p_{copy}(w) + p(z=0) p_{softmax}(w)`
    """

    def __init__(self, dim, generator, eps=1e-20):
        """
        Args:
            dim (int): size of input representation
            generator (nn.Module): map a embedding into word distribution over tgt_vocab
            eps:
        """
        super(CopyAttention, self).__init__()
        self.eps = eps

        self.generator = generator
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_copy = nn.Linear(dim, 1)

    def forward(self, query, memory_bank, stok_mask, src_map):
        """ Compute a distribution over the target dictionary extended by the dynamic
        dictionary implied by copying source words.
        Args:
            query (FloatTensor): hidden outputs `[batch, tgt_len, dim]`
            memory_bank (FloatTensor): global attn `[batch, src_len, dim]`
            src_pad_mask (BoolTensor): `[batch, src_len]`
            src_map (FloatTensor): A sparse indicator matrix mapping each source word to
              its idx in the "extended" vocab. `[batch, src_len, extra_words]`,
              (i,j,k) = 1: j-th tok's attn score will be mapped to dy_vocab[k]
        """
        # original probabilities.
        p_copy = torch.sigmoid(self.linear_copy(query))  # p(z=1), `[batch, tgt_len, 1]`

        prob = self.generator(query)
        # prob[:, :, PAD] = -self.eps  # mask out `PAD` #感觉这个地方不写应该也没啥问题
        prob = F.softmax(prob, dim=-1)  # `[batch, tgt_len, vocab_size]`

        ori_prob = prob * (1 - p_copy)  # p_{word}(w) * (1 - p(z))

        global_attn = self.linear_in(query) @ memory_bank.transpose(-1, -2)  # bs,tgt_l,src_l
        if stok_mask is not None:
            src_pad_mask = stok_mask.unsqueeze(1)  # `[batch, 1, src_len]`
            global_attn = global_attn.masked_fill(src_pad_mask, -1e18)

        copy_prob = F.softmax(global_attn, dim=-1) * p_copy  # `[batch, tgt_len, src_len]`
        copy_prob = copy_prob @ src_map  # `[batch, tgt_len, extra_words]`
        # bs,tgt_l,src_l,   bs,src_l,max_vob_size -> bs,tgt_l,max_vob_size, 得到的是每个vocb上的概率分布
        # 但是问题是，你这个概率和前边的概率可能会有重叠啊，你这样直接拼一块能行吗？
        return torch.cat([ori_prob, copy_prob], dim=2)


def merge_copy_dist(prediction, shared_idxs, beam_size):
    """ merge the normal prediction over tgt vocab and prediction over per-example
    dynamic vocab.
    Args:
        prediction (FloatTensor): the combined prediction dist over extended vocab,
            (tgt_vocab, dynamic_vocab) shape as `[batch, extended_vocab_size]`.
        shared_idxs (List[LongTensor, LongTensor]): the idxs in tgt vocab and dynamic
            vocab of the shared words between this two vocabs.
            - dy_idx (LongTensor): the idxs of shared words in dynamic vocab.
            - tgt_idx (LongTensor): the idxs of shared words in tgt vocab.
        beam_size (int): size of beam search.
    Returns:
        prediction (FloatTensor): shape as `[batch, extended_vocab_size]`
    """
    device = prediction.device
    for batch_idx, (dy_idx, tgt_idx) in enumerate(shared_idxs):  # share idx没有扩张，所以一个share idx管一个beam所有的转换
        if beam_size is not None:
            for b_idx in range(beam_size):
                idx = batch_idx * beam_size + b_idx
                dy_shared_scores = prediction[idx].index_select(0, dy_idx.to(device))
                prediction[idx].index_add_(0, tgt_idx.to(device), dy_shared_scores)
                prediction[idx].index_fill_(0, dy_idx.to(device), 1e-10)
        else:
            dy_shared_scores = prediction[batch_idx].index_select(0, dy_idx.to(device))
            prediction[batch_idx].index_add_(0, tgt_idx.to(device), dy_shared_scores)
            prediction[batch_idx].index_fill_(0, dy_idx.to(device), 1e-10)
            # prediction[batch_idx]是这个样本上的概率分布
            # 这个等于是把copy算出的概率移动到了正常此表上，然后再把copy对应位置上的概率清零
    return prediction
