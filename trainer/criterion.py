from dataset import UNK
import torch


class CopyGeneratorCriterion:
    def __init__(self, vocab_size, eps=1e-20):
        self.offset = vocab_size
        self.eps = eps

    def __call__(self, scores, dy_tgt, tgt):
        '''

        :param scores: `[batch, tgt_len - 1, vocab_size]`
        :param dy_tgt: `[batch, tgt_len - 1]`
        :param tgt: `[batch, tgt_len - 1]`
        :return: [batch * tgt_len - 1]
        '''
        dy_tgt = dy_tgt.view(-1)  # bs*l
        tgt = tgt.view(-1)
        scores = scores.view(-1, scores.size(2))  # bs*l,vocab_size

        dy_tgt_unk = dy_tgt.eq(UNK).float()
        dy_tgt_not_unk = dy_tgt.ne(UNK).float()
        tgt_unk = tgt.eq(UNK).float()
        tgt_not_unk = tgt.ne(UNK).float()

        copy_scores = scores.gather(1, dy_tgt.view(-1, 1) + self.offset).view(-1)
        copy_scores = copy_scores * dy_tgt_not_unk + self.eps  # 如果这个词没有在dy词表里出现，那当然就不指针了

        ori_scores = scores.gather(1, tgt.view(-1, 1)).view(-1)

        final_scores = copy_scores + ori_scores * tgt_not_unk
        final_scores = final_scores + ori_scores * dy_tgt_unk * tgt_unk
        # 就是说，对于那些在原词表中转换成unk的单词来说，只有当这个单词在dy词表里也找不到指的东西，那就还是算原始词表上unk的loss，否则是不用算的
        loss = - final_scores.log()  # bs,l-1
        return loss
