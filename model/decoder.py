import torch
import torch.nn.functional as F
from torch import nn
from .utils import SublayerConnection, PositionwiseFeedForward
import math


class MultiHeadedCrossAttention(nn.Module):
    def __init__(self, hidden, attn_heads, dropout):
        super(MultiHeadedCrossAttention, self).__init__()
        self.KV_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(2)])
        self.Q_linear = nn.Linear(hidden, hidden)
        self.output_linear = nn.Linear(hidden, hidden)
        self.hidden = hidden
        self.d_k = hidden // attn_heads
        assert hidden % attn_heads == 0
        self.dropout = nn.Dropout(p=dropout)
        self.attn_heads = attn_heads

    def forward(self, tgt, memory, mask=None, layer_cache=None):
        batch_size, key_len = memory.shape[:2]

        if layer_cache is not None:
            # means step_wise_decode
            if layer_cache['memory_keys'] is None:
                # means the first step
                key, value = [l(x).view(batch_size, -1, self.attn_heads, self.d_k).transpose(1, 2)
                              for l, x in zip(self.KV_layers, (memory, memory))]
                layer_cache["memory_keys"] = key  # bs,h,l,head
                layer_cache["memory_values"] = value
            else:
                # means not the first step, so no need re-calculate
                key, value = layer_cache["memory_keys"], layer_cache["memory_values"]
        else:
            # means decode in train
            key, value = [l(x).view(batch_size, -1, self.attn_heads, self.d_k).transpose(1, 2)
                          for l, x in zip(self.KV_layers, (memory, memory))]
        query = self.Q_linear(tgt).view(batch_size, -1, self.attn_heads, self.d_k).transpose(1, 2)
        query = query / math.sqrt(self.d_k)
        attn_score = query @ key.transpose(-1, -2)
        if mask is not None:
            mask = mask.unsqueeze(1)  # # bs,1,l -> `[batch, 1, 1, seq_len]`
            attn_score = attn_score.masked_fill(mask, -1e18)
        attn = self.dropout(F.softmax(attn_score, dim=-1))
        context = attn @ value  # `[batch, head, query_len, head_dim]`
        context = context.transpose(1, 2).reshape(batch_size, -1, self.hidden)
        final_output = self.output_linear(context)

        return final_output


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, hidden, attn_heads, dropout):
        super(MultiHeadedSelfAttention, self).__init__()
        self.input_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
        self.output_linear = nn.Linear(hidden, hidden)
        self.hidden = hidden
        self.d_k = hidden // attn_heads
        assert hidden % attn_heads == 0
        self.dropout = nn.Dropout(p=dropout)
        self.attn_heads = attn_heads

    def forward(self, tgt, mask=None, layer_cache=None):
        batch_size = tgt.shape[0]
        if layer_cache is not None:
            # means step_wise_decode
            query, key, value = [l(x).view(batch_size, -1, self.attn_heads, self.d_k).transpose(1, 2)
                                 for l, x in zip(self.input_layers, (tgt, tgt, tgt))]

            if layer_cache['self_keys'] is not None:
                # remember the length of query is 1,
                # it means not first step, so should concat new key to old keys
                key = torch.cat([layer_cache["self_keys"], key], dim=2)  # bs,h,l,d
            if layer_cache["self_values"] is not None:
                value = torch.cat([layer_cache["self_values"], value], dim=2)
            layer_cache["self_keys"] = key
            layer_cache["self_values"] = value
        else:
            # means decode in train
            query, key, value = [l(x).view(batch_size, -1, self.attn_heads, self.d_k).transpose(1, 2)
                                 for l, x in zip(self.input_layers, (tgt, tgt, tgt))]

        query = query / math.sqrt(self.d_k)
        attn_score = query @ key.transpose(-1, -2)
        if mask is not None:
            mask = mask.unsqueeze(1)  # # bs,1,l -> `[batch, 1, 1, seq_len]`
            attn_score = attn_score.masked_fill(mask, -1e18)
        attn = self.dropout(F.softmax(attn_score, dim=-1))
        context = attn @ value  # `[batch, head, query_len, head_dim]`
        context = context.transpose(1, 2).reshape(batch_size, -1, self.hidden)
        final_output = self.output_linear(context)
        return final_output


class TransformerDecoderBlock(nn.Module):
    def __init__(self, args):
        super(TransformerDecoderBlock, self).__init__()

        self.self_sublayer = SublayerConnection(args.hidden, args.dropout, args.decoder_pre_norm)
        self.cross_sublayer = SublayerConnection(args.hidden, args.dropout, args.decoder_pre_norm)
        self.out_sublayer = SublayerConnection(args.hidden, args.dropout, args.decoder_pre_norm)

        self.cross_attn = MultiHeadedCrossAttention(args.hidden, args.attn_heads, args.dropout)
        self.self_attn = MultiHeadedSelfAttention(args.hidden, args.attn_heads, args.dropout)
        self.feed_forward = PositionwiseFeedForward(args.hidden, args.d_ff_fold, args.dropout, args.activation)

    def forward(self, tgt, memory, src_pad_mask, tgt_mask, layer_cache=None):
        tgt = self.self_sublayer(tgt, lambda x: self.self_attn(x, mask=tgt_mask, layer_cache=layer_cache))
        tgt = self.cross_sublayer(tgt, lambda x: self.cross_attn(x, memory=memory, mask=src_pad_mask,
                                                                 layer_cache=layer_cache))
        tgt = self.out_sublayer(tgt, self.feed_forward)
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, args):
        super(TransformerDecoder, self).__init__()
        self.args = args
        self.transformer_blocks = nn.ModuleList([TransformerDecoderBlock(args) for _ in range(args.decoder_layers)])
        self.attn_heads = args.attn_heads
        self.d_k = args.hidden // args.attn_heads
        self.num_layers = args.decoder_layers

    def init_cache(self, cache):
        for i in range(self.num_layers):
            layer_cache = {"memory_keys": None, "memory_values": None, "self_keys": None, "self_values": None}
            cache[f'layer_{i}'] = layer_cache

    @staticmethod
    def add_future_mask(tgt_src_pad_mask):
        """ add the future mask for tgt_src
        Args:
            tgt_src_pad_mask (BoolTensor): shape as `[batch, 1, tgt_src_len]`
        Returns:
            BoolTensor:
            - tgt_src_mask, shape as `[batch, tgt_src_len, tgt_src_len]`
        """
        tgt_len = tgt_src_pad_mask.size(-1)
        future_mask = torch.ones([tgt_len, tgt_len], dtype=torch.uint8)
        future_mask = future_mask.to(tgt_src_pad_mask.device)
        future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
        return torch.gt(tgt_src_pad_mask + future_mask, 0)

    def forward(self, memory, src_pad_mask, emb, tgt_pad_mask, cache=None, step=None):
        if step == 0:
            # means step_wise_decode, and should init cache, and cache = {}
            self.init_cache(cache)
        tgt_pad_mask = tgt_pad_mask.unsqueeze(1)  # bs,1,1 for step_wise_decode
        if step is None:
            # means decode, so tgt_pad_mask should convert by adding future mask
            tgt_mask = self.add_future_mask(tgt_pad_mask)  # 1 means should been masked
        else:
            tgt_mask = tgt_pad_mask
        src_pad_mask = src_pad_mask.unsqueeze(1)  # bs,1,l

        out = emb
        for i, layer in enumerate(self.transformer_blocks):
            layer_cache = cache[f"layer_{i}"] if cache is not None else None
            out = layer(out, memory, src_pad_mask, tgt_mask, layer_cache)

        return out
