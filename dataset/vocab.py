import os
import json
import unicodedata
from collections.abc import Iterable
from collections import Counter

PAD, PAD_WORD = 0, '<PAD>'
UNK, UNK_WORD = 1, '<UNK>'
BOS, BOS_WORD = 2, '<BOS>'
EOS, EOS_WORD = 3, '<EOS>'


class Vocab(object):
    def __init__(self, words=None, no_special_token=False):
        if no_special_token:
            self.tok2ind = {PAD_WORD: PAD,
                            UNK_WORD: UNK}
            self.ind2tok = {PAD: PAD_WORD,
                            UNK: UNK_WORD}
        else:
            self.tok2ind = {PAD_WORD: PAD,
                            UNK_WORD: UNK,
                            BOS_WORD: BOS,
                            EOS_WORD: EOS}
            self.ind2tok = {PAD: PAD_WORD,
                            UNK: UNK_WORD,
                            BOS: BOS_WORD,
                            EOS: EOS_WORD}

        if words is not None:
            self.add_words(words)
        self.pad_index = PAD

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def add_words(self, words):
        assert isinstance(words, Iterable)
        for word in words:
            self.add(word)

    def add(self, word):
        word = self.normalize(word)
        if word not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[word] = index
            self.ind2tok[index] = word

    def __len__(self):
        return len(self.tok2ind)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.ind2tok.get(key, UNK_WORD)
        elif isinstance(key, str):
            return self.tok2ind.get(self.normalize(key), self.tok2ind.get(UNK_WORD))
        else:
            raise RuntimeError('Invalid key type.')


def create_vocab(dataset, only_train, end, vocab_size, no_special_token, ):
    train_path = './data/{}/{}_{}_dic.json'.format(dataset, 'train', end)

    with open(os.path.join(train_path), 'r') as f:
        vocab_dict = json.load(f)

    norm_dict = dict()
    for key, val in vocab_dict.items():
        norm_key = Vocab.normalize(key)
        if norm_key in norm_dict:
            norm_dict[norm_key] += val
        else:
            norm_dict[norm_key] = val
    if not only_train:
        test_path = './data/{}/{}_{}_dic.json'.format(dataset, 'test', end)
        with open(os.path.join(test_path), 'r') as f:
            vocab_dict = json.load(f)
        for key, val in vocab_dict.items():
            norm_key = Vocab.normalize(key)
            if norm_key in norm_dict:
                norm_dict[norm_key] += val
            else:
                norm_dict[norm_key] = val
    word_count = Counter(norm_dict)
    most_common = word_count.most_common(vocab_size)
    word_set = [word for word, _ in most_common]
    vocab = Vocab(word_set, no_special_token)
    return vocab
