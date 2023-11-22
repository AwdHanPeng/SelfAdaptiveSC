from torch.utils.data import Dataset, Sampler
import os
import torch
import random
from tqdm import tqdm
# import joblib
import pickle as joblib
from .vocab import Vocab, UNK, EOS, BOS, BOS_WORD, EOS_WORD
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.nn.functional as F


class SCDataset(Dataset):
    def __init__(self, args, s_vocab, t_vocab, split):
        self.args = args
        self.s_vocab = s_vocab
        self.t_vocab = t_vocab
        assert split in ['train', 'test']  # no valid
        self.split = split
        dataset_dir = os.path.join('./data', args.dataset)
        self.pkl_path = os.path.join(dataset_dir, split + '.pkl')
        self.offset_path = os.path.join(dataset_dir, split + '_offset.pkl')

        with open(self.offset_path, 'rb') as f:
            self.offset = joblib.load(f)

        self.corpus_line = len(self.offset)
        if self.args.tiny_data > 0:
            self.corpus_line = self.args.tiny_data
        print('Loading {} off Memory'.format(split))

    def __len__(self):
        return self.corpus_line

    def __getitem__(self, item):
        sample = self.process(self.get_data(item))

        return sample

    def get_data(self, item):
        with open(self.pkl_path, 'rb') as f:
            f.seek(self.offset[item])
            data = joblib.load(f)
        return data

    def find_shared_words(self, dynamic_vocab):
        offset = len(self.t_vocab)
        shared_dy_idxs, shared_tgt_idxs = [], []
        for i in range(2, len(dynamic_vocab)):
            src_word = dynamic_vocab[i]
            tgt_idx = self.t_vocab[src_word]
            if tgt_idx != UNK:
                shared_dy_idxs.append(offset + i)
                shared_tgt_idxs.append(tgt_idx)
        return torch.LongTensor(shared_dy_idxs), torch.LongTensor(shared_tgt_idxs)

    @staticmethod
    def actions_process(actions, max_depth, max_ary):

        def my_abs(length):
            return length if length >= 0 else 0

        sample_idx = []
        pad_idx = 0

        for node_action in actions:
            sample_idx.append(
                [action if action < max_ary else max_ary for action in node_action][:max_depth] + [pad_idx] * my_abs(
                    max_depth - len(node_action)))
        return torch.tensor(sample_idx, dtype=torch.long)

    def process(self, data):
        '''
        data = {'target': data['summary_seq'], 'input': all_tokens, 'node_actions': node_actions, 'leaf_idx': leaf_idx,
        'father_actions': father_actions, }
        :param data:
        :return:
        '''
        example = dict()
        assert len(data['input']) == len(data['leaf_idx']) == len(data['node_actions'])

        leaf_tokens = [token for token, leaf_id in zip(data['input'], data['leaf_idx']) if leaf_id != 0]
        dynamic_vocab = Vocab(leaf_tokens, no_special_token=True)
        example['dynamic_vocab'] = dynamic_vocab

        tgt_seq = [BOS_WORD] + data['target'] + [EOS_WORD]
        example['shared_idxs'] = self.find_shared_words(dynamic_vocab)
        example['stok_seq_dy_rep'] = torch.tensor([dynamic_vocab[w] for w in data['input']], dtype=torch.long)
        example['tgt_seq_dy_rep'] = torch.tensor([dynamic_vocab[w] for w in tgt_seq], dtype=torch.long)

        example['stok_seq_rep'] = torch.tensor([self.s_vocab[w] for w in data['input']], dtype=torch.long)
        example['tgt_seq_rep'] = torch.tensor([self.t_vocab[w] for w in tgt_seq], dtype=torch.long)

        example['leaf_idx'] = torch.tensor(data['leaf_idx'], dtype=torch.long)

        example['node_actions'] = self.actions_process(data['node_actions'], self.args.max_depth, self.args.max_ary)
        example['father_actions'] = self.actions_process(data['father_actions'], self.args.max_depth, self.args.max_ary)

        example['tgt_seq'] = data['target']  # no BOS and EOS
        example['leaf_seq'] = leaf_tokens  # 0 means non-leaf, else means leaf idx
        return example

    @staticmethod
    def collect_fn(batch):
        bs = len(batch)
        dynamic_vocabs, src_maps, shared_idxs = [], [], []
        tgt_seq_reps, tgt_lens, tgt_seq_dy_reps = [], [], []
        stok_seq_reps = []
        node_actions, father_actions = [], []
        leaf_idx = []

        for ex in batch:
            # code information
            stok_seq_reps.append(ex['stok_seq_rep'])

            # summary information
            tgt_seq_reps.append(ex['tgt_seq_rep'])
            tgt_lens.append(len(ex['tgt_seq_rep']))  # use the length of ADD BOS and EOS

            # for copy attn
            dynamic_vocabs.append(ex['dynamic_vocab'])
            src_maps.append(ex['stok_seq_dy_rep'])
            shared_idxs.append(ex['shared_idxs'])
            tgt_seq_dy_reps.append(ex['tgt_seq_dy_rep'])

            node_actions.append(ex['node_actions'])
            father_actions.append(ex['father_actions'])
            leaf_idx.append(ex['leaf_idx'])

        stok_seq_rep = pad_sequence(stok_seq_reps, batch_first=True, padding_value=0)
        tgt_seq_rep = pad_sequence(tgt_seq_reps, batch_first=True, padding_value=0)
        tgt_seq_len = torch.LongTensor(tgt_lens)
        src_map = pad_sequence(src_maps, batch_first=True, padding_value=0)
        tgt_seq_dy_rep = pad_sequence(tgt_seq_dy_reps, batch_first=True, padding_value=0)

        node_action = pad_sequence(node_actions, batch_first=True, padding_value=0)  # bs,l,depth
        father_action = pad_sequence(father_actions, batch_first=True, padding_value=0)
        leaf_idx = pad_sequence(leaf_idx, batch_first=True, padding_value=0)
        return {
            'batch_size': bs,
            'stok_seq_rep': stok_seq_rep,
            'tgt_seq_rep': tgt_seq_rep,
            'tgt_seq_len': tgt_seq_len,
            'src_text': [" ".join(ex['leaf_seq']) for ex in batch],
            'tgt_text': [" ".join(ex['tgt_seq']) for ex in batch],
            'dynamic_vocabs': dynamic_vocabs,
            'src_map': F.one_hot(src_map).float(),  # `[batch, src_len, extra_words]` 每个位置上是这个单词在dy词表里的idx，并onehot
            'tgt_seq_dy_rep': tgt_seq_dy_rep,
            'shared_idxs': shared_idxs,
            'node_action': node_action,
            'father_action': father_action,
            'leaf_idx': leaf_idx
        }


class LengthGroupSampler(Sampler):
    def __init__(self, args, split, batch_size, mode='train'):
        self.args = args
        self.dataset_dir = os.path.join('./data', args.dataset)
        self.pkl_path = os.path.join(self.dataset_dir, split + '.pkl')

        assert split in ['train', 'test']
        self.batch_size = batch_size

        self.lengths_path = os.path.join(self.dataset_dir, split + '_lengths.pkl')

        with open(self.lengths_path, 'rb') as f:
            self.lengths = joblib.load(f)
        self.corpus_line = len(self.lengths)
        if self.args.tiny_data > 0:
            self.corpus_line = self.args.tiny_data
            self.lengths = self.lengths[:self.args.tiny_data]
        self.mode = mode

    def __iter__(self):
        if self.mode == 'train':
            lengths = np.array(
                [(-l[0], -l[1], np.random.random()) for l in self.lengths],
                dtype=[('l1', np.int_), ('l2', np.int_), ('rand', np.float_)]
            )
            indices = np.argsort(lengths, order=('l1', 'l2', 'rand'))
            batches = [indices[i:i + self.batch_size]
                       for i in range(0, len(indices), self.batch_size)]
            np.random.shuffle(batches)

        elif self.mode == 'test':
            lengths = np.array(
                [(-l[0], -l[1]) for l in self.lengths],
                dtype=[('l1', np.int_), ('l2', np.int_), ]
            )
            indices = np.argsort(lengths, order=('l1', 'l2'))
            batches = [indices[i:i + self.batch_size]
                       for i in range(0, len(indices), self.batch_size)]
        else:
            raise Exception('No Valid Mode')

        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return self.corpus_line
