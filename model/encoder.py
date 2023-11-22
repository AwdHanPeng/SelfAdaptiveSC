import math

import torch
from torch.nn import init
from torch import nn
from .utils import SublayerConnection, PositionwiseFeedForward, relative_position_bucket, init_embedding_weights, \
    RoFormerSinusoidalPositionalEmbedding
import torch.nn.functional as F


class TreePositionalEncoding(nn.Module):
    def __init__(self, args):
        super(TreePositionalEncoding, self).__init__()
        self.max_ary = args.max_ary
        self.hidden = args.hidden
        self.length = self.max_ary * (1 + self.max_ary) // 2 + 1
        self.embedding = nn.Embedding(self.length, args.action_size, padding_idx=0)
        init_embedding_weights(self.embedding)
        self.linear = nn.Linear(args.max_depth * args.action_size, args.hidden)
        self.norm = nn.LayerNorm(args.hidden)
        self.QK_linear = nn.ModuleList([nn.Linear(self.hidden, self.hidden) for _ in range(2)])
        self.d_k = args.hidden // args.attn_heads
        self.attn_heads = args.attn_heads

    def forward(self, node_action, father_action):
        batch_size, length = node_action.shape[:2]
        idx = node_action + (father_action * (father_action - 1) / 2).to(torch.long)
        p_embedding = self.embedding(idx).view(batch_size, length, -1)
        p_embedding = self.norm(self.linear(p_embedding))  # bs,l,hidden
        p_query, p_key = [l(x).view(batch_size, -1, self.attn_heads, self.d_k).transpose(1, 2)
                          for l, x in zip(self.QK_linear, (p_embedding, p_embedding))]
        p_query = p_query / math.sqrt(self.d_k)
        score = p_query @ p_key.transpose(-1, -2)
        return score


class LeafPositionalEncoding(nn.Module):
    def __init__(self, args):
        super(LeafPositionalEncoding, self).__init__()
        self.p = nn.Embedding(args.max_source_len, args.hidden, padding_idx=0)
        init_embedding_weights(self.p)
        # padding as num_buckets, for non-t vs. non-t
        self.r = nn.Embedding(args.num_buckets + 1, args.attn_heads, padding_idx=args.num_buckets)
        init_embedding_weights(self.r)
        self.leaf_PE_Type = args.leaf_PE_Type
        self.hidden = args.hidden
        self.attn_heads = args.attn_heads
        self.QK_linear = nn.ModuleList([nn.Linear(self.hidden, self.hidden) for _ in range(2)])
        self.d_k = args.hidden // args.attn_heads
        self.args = args
        self.norm = nn.LayerNorm(args.hidden)

    @staticmethod
    def create_t5_score(leaf_idx, num_buckets):
        # leaf_idx: bs,l
        relative_position = leaf_idx.unsqueeze(1) - leaf_idx.unsqueeze(-1)
        relative_bucket = relative_position_bucket(relative_position, num_buckets=num_buckets)  # bs,l,l

        leaf_mask = leaf_idx != 0  # non-leaf->False; leaf->True, # bs,l
        leaf_mask = leaf_mask.unsqueeze(1) * leaf_mask.unsqueeze(-1)  # bs,l,l, leaf*leaf->True
        leaf_mask = leaf_mask == 0  # leaf*leaf->False; bs,l,l

        # convert others as padding_idx
        relative_bucket = relative_bucket.masked_fill(leaf_mask, num_buckets)
        return relative_bucket

    def forward(self, leaf_idx):
        batch_size, length = leaf_idx.shape[:2]
        t5_score = self.r(self.create_t5_score(leaf_idx, self.args.num_buckets)).permute(0, 3, 1, 2)
        # bs,l,l -> bs,l,l,h -> bs,h,l,l

        p_embedding = self.norm(self.p(leaf_idx))
        p_query, p_key = [l(x).view(batch_size, -1, self.attn_heads, self.d_k).transpose(1, 2)
                          for l, x in zip(self.QK_linear, (p_embedding, p_embedding))]
        p_query = p_query / math.sqrt(self.d_k)
        untied_score = p_query @ p_key.transpose(-1, -2)  # bs,h,l,l

        if self.leaf_PE_Type == 'Merge':
            score = untied_score + t5_score
        elif self.leaf_PE_Type == 'T5':
            score = t5_score

        else:
            score = untied_score
        return score


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadedSelfAttention, self).__init__()
        hidden = args.hidden
        attn_heads = args.attn_heads
        self.input_layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
        self.output_linear = nn.Linear(hidden, hidden)
        self.hidden = hidden
        self.attn_heads = attn_heads
        self.d_k = hidden // attn_heads
        assert hidden % attn_heads == 0
        self.dropout = nn.Dropout(args.dropout)
        self.print_avg_gate = args.print_avg_gate
        self.merge_type = args.type
        assert self.merge_type in ['sentinel', 'tree', 'add', 'none', 'leaf']
        self.sentinel_linear = nn.Linear(self.hidden, self.attn_heads)

        self.roformer = RoFormerSinusoidalPositionalEmbedding(args.max_source_len, args.hidden // args.attn_heads)
        self.args = args
        if self.args.sentinel_on_key:
            self.sentinel_weight = nn.Parameter(torch.rand(self.attn_heads, self.d_k))
            if self.args.new_query:
                self.query_linear = nn.Linear(hidden, hidden)

    @staticmethod
    def create_leaf_mask(leaf_idx):  # bs,l
        leaf_mask = leaf_idx != 0  # True for leaf, False for non-leaf
        return leaf_mask.unsqueeze(1) * leaf_mask.unsqueeze(-1)  # bs,l,l leaf*leaf->True, else False

    @staticmethod
    def apply_rotary(x, sinusoidal_pos):
        sin, cos = sinusoidal_pos
        x1, x2 = x[..., 0::2], x[..., 1::2]
        # 如果是旋转query key的话，下面这个直接cat就行，因为要进行矩阵乘法，最终会在这个维度求和。（只要保持query和key的最后一个dim的每一个位置对应上就可以）
        # torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        # 如果是旋转value的话，下面这个stack后再flatten才可以，因为训练好的模型最后一个dim是两两之间交替的。
        return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2, -1)

    def forward(self, content, mask, leaf_idx, tree_score, leaf_score):
        batch_size = content.shape[0]
        query, key, value = [l(x).view(batch_size, -1, self.attn_heads, self.d_k).transpose(1, 2)
                             for l, x in zip(self.input_layers, (content, content, content))]
        # bs,h,l,d
        query = query / math.sqrt(self.d_k)
        attn_score = query @ key.transpose(-1, -2)
        mask = mask.unsqueeze(1).repeat(1, mask.shape[-1], 1).unsqueeze(1)  # bs,l -> bs,l,l -> bs,1,l,l

        if self.merge_type == 'sentinel':

            # for tree
            tree_merge_score = (attn_score + tree_score) / math.sqrt(2)
            tree_merge_score = tree_merge_score.masked_fill(mask, -1e18)
            tree_attn = F.softmax(tree_merge_score, dim=-1)  # bs,h,l,l

            # for leaf

            if self.args.rot:
                sinusoidal_pos = self.roformer(leaf_idx).unsqueeze(1).chunk(2, dim=-1)
                # bs,l -> bs,l,d_k  -> bs,1,l,d_k -> (bs,1,l,d_k/2, bs,1,l,d_k/2, )
                temp_query = self.apply_rotary(query, sinusoidal_pos)
                temp_key = self.apply_rotary(key, sinusoidal_pos)
                leaf_merge_score = temp_query @ temp_key.transpose(-1, -2)
            else:
                leaf_merge_score = (attn_score + leaf_score) / math.sqrt(2)
            leaf_mask = self.create_leaf_mask(leaf_idx)
            leaf_merge_score = leaf_merge_score.masked_fill(leaf_mask.unsqueeze(1) == 0, -1e18)
            leaf_attn = F.softmax(leaf_merge_score, dim=-1)  # bs,h,l,l

            if self.args.sentinel_on_key:
                if self.args.new_query:
                    new_query = self.query_linear(content).view(batch_size, -1, self.attn_heads, self.d_k)
                    # bs,l,h,d
                    gate_attn = torch.einsum('blhd,hd->blh', new_query, self.sentinel_weight)
                else:
                    gate_attn = torch.einsum('blhd,hd->blh', query.transpose(1, 2), self.sentinel_weight)
            else:
                gate_attn = torch.sigmoid(self.sentinel_linear(content))  # bs,l,head
            gate_attn = gate_attn.masked_fill((leaf_idx == 0).unsqueeze(-1), 1).transpose(-1, -2).unsqueeze(-1)
            # bs,l,head -> bs,head,l -> bs,head,l,1
            # leaf_idx = 0 means non-leaf, so attn always = 1 =>bs,l,1
            attn = tree_attn * gate_attn + leaf_attn * (1 - gate_attn)

            if self.print_avg_gate:  # bs,head,l,1 -> bs,head,l -> bs,l,head
                # leaf_idx bs,l
                masked_gate_attn = gate_attn.squeeze(-1).transpose(-1, -2).masked_fill((leaf_idx == 0).unsqueeze(-1), 0)
                masked_gate_attn = torch.mean(masked_gate_attn, dim=-1, keepdim=False)  # bs,l,head -> bs,l
                gate_attn_sum = torch.sum(masked_gate_attn, dim=-1, keepdim=False)  # bs,l -> bs
                gate_attn_avg = gate_attn_sum / torch.count_nonzero(leaf_idx, dim=-1)
                print('Average Gate Prob: {} '.format(torch.mean(gate_attn_avg).item()))
                # 注意，这个地方计算的是一个batch的样本在某一层的概率均值，我觉得这个东西观察一下就可以了，训练的时候没必要监控了

        elif self.merge_type == 'add':
            leaf_mask = self.create_leaf_mask(leaf_idx)
            merge_score = (attn_score + tree_score + leaf_score.masked_fill(leaf_mask.unsqueeze(1) == 0,
                                                                            0)) / math.sqrt(3)
            merge_score = merge_score.masked_fill(mask, -1e18)
            attn = F.softmax(merge_score, dim=-1)
        elif self.merge_type == 'tree':
            tree_merge_score = (attn_score + tree_score) / math.sqrt(2)
            tree_merge_score = tree_merge_score.masked_fill(mask, -1e18)
            attn = F.softmax(tree_merge_score, dim=-1)
        elif self.merge_type == 'leaf':
            if self.args.rot:
                sinusoidal_pos = self.roformer(leaf_idx).unsqueeze(1).chunk(2, dim=-1)
                # bs,l -> bs,l,d_k  -> bs,1,l,d_k -> (bs,1,l,d_k/2, bs,1,l,d_k/2, )
                temp_query = self.apply_rotary(query, sinusoidal_pos)
                temp_key = self.apply_rotary(key, sinusoidal_pos)
                leaf_merge_score = temp_query @ temp_key.transpose(-1, -2)
                # 如果是rot的话，这样计算就已经是带有contextAttn和peBias的结果了
            else:
                leaf_mask = self.create_leaf_mask(leaf_idx)
                leaf_merge_score = (attn_score + leaf_score.masked_fill(leaf_mask.unsqueeze(1) == 0, 0)) / math.sqrt(2)
                # 这个操作的意思是，如果只要leaf的话，attnscore不变，然后leafscore部分只保留leaf2leaf的score，其他部分都完全置为0
                # 这样就不会影响attnscore的数值
            merge_score = leaf_merge_score.masked_fill(mask, -1e18)
            # 然后这个部分的mask，则是为了处理padding的mask
            attn = F.softmax(merge_score, dim=-1)
        else:
            attn = F.softmax(attn_score.masked_fill(mask, -1e18), dim=-1)

        content = self.dropout(attn) @ value  # bs,h,l,d
        content = content.transpose(1, 2).reshape(batch_size, -1, self.hidden)
        return self.output_linear(content)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, args):
        super(TransformerEncoderBlock, self).__init__()
        self.input_sublayer = SublayerConnection(args.hidden, args.dropout, args.pre_norm)
        self.output_sublayer = SublayerConnection(args.hidden, args.dropout, args.pre_norm)
        self.self_attn = MultiHeadedSelfAttention(args)
        self.feed_forward = PositionwiseFeedForward(args.hidden, args.e_ff_fold, args.dropout, args.activation)

    def forward(self, content, mask, leaf_idx, tree_score, leaf_score):
        content = self.input_sublayer(content,
                                      lambda x: self.self_attn(x, mask, leaf_idx, tree_score, leaf_score))
        content = self.output_sublayer(content, self.feed_forward)
        return content


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.transformer_blocks = nn.ModuleList([TransformerEncoderBlock(args) for _ in range(args.encoder_layers)])
        self.h = args.attn_heads
        self.d_k = args.hidden // args.attn_heads
        self.hidden = args.hidden
        self.num_layers = args.encoder_layers
        self.tree_positional_encoding = TreePositionalEncoding(args)
        self.leaf_positional_encoding = LeafPositionalEncoding(args)

    def forward(self, emb, mask, node_action, father_action, leaf_idx):
        leaf_score = self.leaf_positional_encoding(leaf_idx)
        tree_score = self.tree_positional_encoding(node_action, father_action)
        content = emb
        for transformer in self.transformer_blocks:
            content = transformer(content, mask, leaf_idx, tree_score, leaf_score)
        return content
