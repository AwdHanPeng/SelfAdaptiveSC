import torch
import torch.nn.functional as F
from torch import nn
from .embedding import SrcEmbedding, TgtEmbedding
from .encoder import TransformerEncoder as Encoder
from .decoder import TransformerDecoder as Decoder
from .pointer import CopyAttention, merge_copy_dist


class SCTransformer(nn.Module):
    def __init__(self, args, s_vocab, t_vocab):
        super(SCTransformer, self).__init__()
        self.args = args
        self.src_embedding = SrcEmbedding(args, s_vocab)
        self.tgt_embedding = TgtEmbedding(args, t_vocab)
        self.t_vocab = t_vocab
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.generator = nn.Linear(args.hidden, len(t_vocab))
        self.copy_attn = CopyAttention(args.hidden, self.generator)

        if args.weight_tying:  # 分类器的权重和embedding互相share
            self.generator.weight = self.tgt_embedding.embedding.weight

    def forward(self, data):
        stok_seq_rep, stok_pad_mask = self.encode(data)

        tgt_seq_rep = data['tgt_seq_rep']
        tgt_pad_mask = tgt_seq_rep == 0
        tgt_emb = self.tgt_embedding(tgt_seq_rep)
        dec_out = self.decoder(stok_seq_rep, stok_pad_mask, tgt_emb, tgt_pad_mask)

        # means no need to pointer non-leaf tokens
        src_map = data['src_map']  # `[batch, src_len, extra_words]`
        leaf_mask = data['leaf_idx'] == 0  # [bs,l], 0 means non-leaf, else means leaf idx

        if self.args.pointer_leaf_mask:
            stok_mask = torch.gt(stok_pad_mask + leaf_mask, 0)
        else:
            stok_mask = torch.gt(stok_pad_mask, 0)

        # stok_mask = torch.gt(stok_pad_mask + leaf_mask, 0)  # bs,l

        scores = self.copy_attn(dec_out, stok_seq_rep, stok_mask, src_map)
        # 注意，这个地方不用对copy的分布进行merge,loss部分是可计算的
        scores = scores[:, :-1, :].contiguous()  # 长度减一，输入为EOS的不用再输出了
        return scores

    def encode(self, data):
        stok_seq_rep = data['stok_seq_rep']  # bs,l
        node_action = data['node_action']
        father_action = data['father_action']
        leaf_idx = data['leaf_idx']
        stok_pad_mask = stok_seq_rep == 0  # bs,l

        stok_seq_emb = self.src_embedding(stok_seq_rep)
        stok_seq_rep = self.encoder(stok_seq_emb, stok_pad_mask, node_action, father_action, leaf_idx)
        return stok_seq_rep, stok_pad_mask

    def step_wise_decode(self, tgt_rep, src_map, shared_idxs, cache, step, memory, memory_key_padding_mask, leaf_idx,
                         beam_size=None, beam_idx=None):
        tgt_rep = tgt_rep.unsqueeze(1)  # bs,1
        tgt = self.tgt_embedding(tgt_rep, step=step)
        tgt_pad_mask = tgt_rep == self.t_vocab.pad_index

        # 开始的时候 beam idx是none 随后通过计算，cache得到初始化，随后beamidx不是none，就可以加载和更新了
        if beam_idx is not None:
            for i in range(self.decoder.num_layers):
                layer_cache = cache[f'layer_{i}']
                layer_cache["memory_keys"] = layer_cache["memory_keys"][beam_idx]
                layer_cache['memory_values'] = layer_cache['memory_values'][beam_idx]
                layer_cache["self_keys"] = layer_cache["self_keys"][beam_idx]
                layer_cache["self_values"] = layer_cache["self_values"][beam_idx]
        # -> `[batch, 1, dim]`

        dec_out = self.decoder(memory, memory_key_padding_mask, tgt, tgt_pad_mask, cache, step)
        # `[batch, 1, tgt_vocab_size + extra_words]`

        # means no need to pointer non-leaf tokens
        leaf_mask = leaf_idx == 0  # [bs,l], 0 means non-leaf, else means leaf idx
        if beam_size is not None:
            leaf_mask = leaf_mask.unsqueeze(1).repeat(1, beam_size, 1).view(-1, leaf_mask.shape[-1])
        if self.args.pointer_leaf_mask:
            stok_mask = torch.gt(memory_key_padding_mask + leaf_mask, 0)
        else:
            stok_mask = torch.gt(memory_key_padding_mask, 0)
        prediction = self.copy_attn(dec_out, memory, stok_mask, src_map)
        prediction = prediction.squeeze(1)  # `[batch, tgt_vocab_size + extra_words]`
        prediction = merge_copy_dist(prediction, shared_idxs, beam_size)

        return prediction
