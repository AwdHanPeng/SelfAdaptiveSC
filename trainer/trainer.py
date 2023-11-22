import torch
from torch import nn
from torch.optim import Adam, AdamW, SGD  # 1.10才有
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from torch.optim.lr_scheduler import MultiStepLR
from dataset import PAD, BOS, UNK, EOS
import math
from .criterion import CopyGeneratorCriterion
from evaluation import calc_metrics
import shutil
from .beam_search import BeamSearchScorer


class Trainer:
    def __init__(self, args, model, train_data, test_infer_data, t_vocab, writer_path):
        self.args = args

        cuda_condition = torch.cuda.is_available() and self.args.with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # 多卡的事情暂时先不考虑
        if cuda_condition and torch.cuda.device_count() > 1:
            self.wrap = True
            model = nn.DataParallel(model)
        else:
            self.wrap = False

        self.model = model.to(self.device)

        self.train_data = train_data
        self.test_infer_data = test_infer_data
        self.optim = AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        # lr=0.0002 weight_decay=0.03 batch_suze=64 num_workers=16
        # grad_norm = clip_grad_norm_(self.model.parameters(), self.config.optim.clip_grad)
        self.milestones = [16, 24, self.args.milestones]
        self.scheduler = MultiStepLR(optimizer=self.optim, milestones=self.milestones)
        # milestones=[16, 24, 28]
        # engine.no_improvement_epochs > config.train.early_stop：8 epoch=30

        self.no_improvement_epochs = 0
        self.warmup_steps = self.args.warmup_steps
        self.warmup_factor = self.args.lr / self.warmup_steps

        self.writer_path = writer_path
        print(self.writer_path)
        self.tensorboard_writer = SummaryWriter(os.path.join('run', self.writer_path))
        self.writer = open(os.path.join('run', self.writer_path, 'experiment.txt'), 'w')
        print(self.args, file=self.writer, flush=True)
        self.iter, self.update_steps = 0, 0
        self.t_vocab = t_vocab
        self.best_epoch, self.best_bleu = 0, float('-inf')
        self.criterion = CopyGeneratorCriterion(len(t_vocab))
        print(
            "Total Parameters: {}*1e6".format(sum([p.nelement() for _, p in self.model.named_parameters()]) // 1e6),
            file=self.writer, flush=True)

        self.accu_steps = self.args.accu_batch_size // self.args.batch_size

    def scheduler_step(self):
        self.scheduler.step()

    def train(self, epoch):
        data_iter = tqdm(enumerate(self.train_data),
                         desc="EP_%s:%d" % ('train', epoch),
                         total=len(self.train_data),
                         bar_format="{l_bar}{r_bar}")
        self.model.train()
        avg_loss = 0.0
        for i, data in data_iter:
            self.iter += 1
            data = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in data.items()}
            scores = self.model(data)
            target = data['tgt_seq_rep'][:, 1:].contiguous()
            tgt_seq_len = data['tgt_seq_len']
            loss = self.criterion(scores, data['tgt_seq_dy_rep'][:, 1:].contiguous(), target)  #
            loss = loss.view(*scores.shape[:-1])  # bs,tgt_len-1
            loss = loss.mul(target.ne(PAD).float()).sum(1)  # bs
            loss_per_token = loss.div((tgt_seq_len - 1).float()).mean().item()  # bs->1
            ppl = math.exp(loss_per_token)
            loss = loss.mean()  # bs->1 每条样本上的loss
            self.tensorboard_writer.add_scalar('Loss', loss, self.iter)
            self.tensorboard_writer.add_scalar('PPL', ppl, self.iter)
            avg_loss += loss.item()
            accu_loss = loss / self.accu_steps
            accu_loss.backward()
            if (i + 1) % self.accu_steps == 0:
                self.update_steps += 1
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                if self.args.warmup:
                    if self.update_steps <= self.warmup_steps:
                        curr_lr = self.warmup_factor * self.update_steps
                        for param_group in self.optim.param_groups:
                            param_group['lr'] = curr_lr
                self.optim.step()
                self.optim.zero_grad()
        print("EP%d_%s, avg_loss=" % (epoch, 'train'), avg_loss / self.iter, file=self.writer, flush=True)
        print('-------------------------------------', file=self.writer, flush=True)
        if self.args.save:
            save_dir = './checkpoint'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(self.model, os.path.join(save_dir, "{}_latest.pth".format(self.writer_path)))

    def greedy_decode(self, memory, memory_key_padding_mask, ex):
        batch_size = memory.shape[0]
        tgt_vocab_size = len(self.t_vocab)
        tgt_rep = torch.full([batch_size], BOS, device=memory.device)  # bs
        acc_gen_seq = tgt_rep.unsqueeze(1).clone()  # bs,1
        cache = {}
        for idx in range(self.args.max_target_len + 1):
            prediction = self.model.step_wise_decode(tgt_rep=tgt_rep, src_map=ex['src_map'],
                                                     shared_idxs=ex['shared_idxs'],
                                                     cache=cache, step=idx, memory=memory,
                                                     memory_key_padding_mask=memory_key_padding_mask,
                                                     leaf_idx=ex['leaf_idx'])
            _, tgt_rep = torch.max(prediction, dim=1)  # bs,
            acc_gen_seq = torch.cat((acc_gen_seq, tgt_rep.unsqueeze(1)), dim=1)
            tgt_rep[tgt_rep >= tgt_vocab_size] = UNK
        return acc_gen_seq

    def beam_search_decode(self, memory, memory_key_padding_mask, ex):
        batch_size = memory.shape[0]
        tgt_vocab_size = len(self.t_vocab)
        beam_size = self.args.width
        batch_beam_size = batch_size * beam_size
        device = memory.device
        beam_scorer = BeamSearchScorer(batch_size, beam_size, device)
        beam_scores = torch.zeros((batch_size, beam_size), dtype=torch.float, device=device)
        beam_scores[:, 1:] = -1e5
        beam_scores = beam_scores.view(-1)

        length = memory.shape[1]
        memory = memory.unsqueeze(1).repeat(1, beam_size, 1, 1).view(batch_beam_size, length, -1)
        memory_key_padding_mask = memory_key_padding_mask.unsqueeze(1). \
            repeat(1, beam_size, 1).view(batch_beam_size, length)
        src_map = ex['src_map'].unsqueeze(1).repeat(1, beam_size, 1, 1).view(batch_beam_size, length, -1)

        tgt_rep = torch.full([batch_beam_size], BOS, dtype=torch.long, device=device)
        acc_gen_rep = tgt_rep.unsqueeze(1).clone()  # l,1
        cache = {}
        beam_idx = None
        for idx in range(self.args.max_target_len + 1):
            prediction = self.model.step_wise_decode(tgt_rep=tgt_rep, src_map=src_map,
                                                     shared_idxs=ex['shared_idxs'],
                                                     cache=cache, step=idx, memory=memory,
                                                     memory_key_padding_mask=memory_key_padding_mask,
                                                     leaf_idx=ex['leaf_idx'], beam_size=beam_size, beam_idx=beam_idx)
            next_token_scores = prediction.log() + beam_scores[:, None]
            vocab_size = next_token_scores.size(-1)
            next_token_scores = next_token_scores.view(batch_size, beam_size * vocab_size)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * beam_size, dim=1, largest=True, sorted=True
            )
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode='floor')
            next_tokens = next_tokens % vocab_size
            beam_outputs = beam_scorer.process(
                acc_gen_rep,  # `[batch * beam_size, curr_len = idx + 1]`
                next_token_scores,  # `[batch, 2 * beam_size]`, values of topK
                next_tokens,  # `[batch, 2 * beam_size]`, idx of topK over vocab
                next_indices,  # `[batch, 2 * beam_size]`, idx of topK over beam
                pad_idx=PAD,
                eos_idx=EOS,
            )
            beam_scores = beam_outputs['next_beam_scores']  # `[batch_beam]` #注意，这个玩意始终是到当前步累计的概率
            tgt_rep = beam_outputs['next_beam_tokens']  # `[batch_beam]`
            beam_idx = beam_outputs['next_beam_indices']  # `[batch_beam]`
            acc_gen_rep = torch.cat((acc_gen_rep[beam_idx], tgt_rep.unsqueeze(1)), dim=-1)
            tgt_rep[tgt_rep >= tgt_vocab_size] = UNK
        pred_seqs = beam_scorer.finalize(
            acc_gen_rep,
            beam_scores,
            PAD, EOS
        )
        return pred_seqs

    def tens2sent(self, preds, ex):
        sentences = []
        for idx, s in enumerate(preds):
            sentence = []
            for wt in s:
                word = wt if isinstance(wt, int) else wt.item()
                if word in [BOS]:
                    continue
                if word in [EOS]:
                    break
                if word < len(self.t_vocab):
                    sentence += [self.t_vocab[word]]
                else:
                    sentence += [ex['dynamic_vocabs'][idx][word - len(self.t_vocab)]]
            if len(sentence) == 0:
                sentence = ['PAD']
            sentence = ' '.join(sentence)
            sentences.append(sentence)
        return sentences

    def predict(self, epoch):
        data_iter = tqdm(enumerate(self.test_infer_data),
                         desc="EP_%s:%d" % ('test', epoch),
                         total=len(self.test_infer_data),
                         bar_format="{l_bar}{r_bar}")
        ref_file_name = os.path.join('run', self.writer_path, 'ref_{}.txt'.format('test'))
        predicted_file_name = os.path.join('run', self.writer_path,
                                           'pred_{}_{}.txt'.format('test', epoch))
        source_file_name = os.path.join('run', self.writer_path, 'source_{}.txt'.format('test'))
        r, p, s = open(ref_file_name, 'w'), open(predicted_file_name, 'w'), open(source_file_name, 'w')
        self.model.eval()
        sources, hypotheses, references = {}, {}, {}
        for i, data in data_iter:
            data = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in data.items()}
            batch_size = data['batch_size']
            memory, memory_key_padding_mask = self.model.encode(data)
            if self.args.beam_search:
                dec_preds = self.beam_search_decode(memory, memory_key_padding_mask, data)
            else:
                dec_preds = self.greedy_decode(memory, memory_key_padding_mask, data)
            prediction = self.tens2sent(dec_preds, data)
            target = data['tgt_text']
            source = data['src_text']
            ex_ids = list(range(i * batch_size, (i * batch_size) + batch_size))

            for key, src, pred, tgt in zip(ex_ids, source, prediction, target):
                hypotheses[key] = [pred]
                references[key] = [tgt]
                sources[key] = src
                r.write(tgt + '\n')
                p.write(pred + '\n')
                s.write(src + '\n')

        result = calc_metrics(hypotheses, references, sources)
        cur_bleu = result['bleu']
        if self.args.train:
            if cur_bleu >= self.best_bleu:
                self.best_epoch = epoch
                self.best_bleu = cur_bleu
                shutil.copyfile('./checkpoint/{}_latest.pth'.format(self.writer_path),
                                './checkpoint/{}_best.pth'.format(self.writer_path))
                print("###Best Result At EP{}, best_bleu={}".format(self.best_epoch, self.best_bleu), file=self.writer,
                      flush=True)
                self.no_improvement_epochs = 0
            else:
                self.no_improvement_epochs += 1
        print(result, file=self.writer, flush=True)
        return result

    def early_stop(self):
        return self.no_improvement_epochs > self.args.early_stop

    def load(self, path):
        self.model = torch.load(path)
        print('Load Pretrain model => {}'.format(path))
        # self.model.eval()
        # self.model = torch.load(path)
        # dic = torch.load(path, map_location='cpu')
        # dic = torch.load(path)
        # for key, _ in dic.items():
        #     if 'module.' in key:
        #         load_pre = 'module.'
        #     else:
        #         load_pre = ''
        #     break
        # for key, _ in self.model.state_dict().items():
        #     if 'module.' in key:
        #         model_pre = 'module.'
        #     else:
        #         model_pre = ''
        #     break
        # if load_pre == '' and model_pre == 'module.':
        #     temp_dict = dict()
        #     for key, value in dic.items():
        #         temp_dict[model_pre + key] = value
        #     dic = temp_dict
        # elif model_pre == '' and load_pre == 'module.':
        #     temp_dict = dict()
        #     for key, value in dic.items():
        #         temp_dict[key.replace(load_pre, model_pre)] = value
        #     dic = temp_dict
        # temp_dict = dict()
        # ori_dic = self.model.state_dict()
        # for key, value in dic.items():
        #     if key in ori_dic and ori_dic[key].shape == value.shape:
        #         temp_dict[key] = value
        # dic = temp_dict
        # for key, value in self.model.state_dict().items():
        #     if key not in dic:
        #         dic[key] = value
        # print([key for key, val in dic.items()])
        # msg = self.model.load_state_dict(dic)
        # print(msg)
