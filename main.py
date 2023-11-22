import argparse
import os.path

from torch.utils.data import DataLoader
from dataset import SCDataset, LengthGroupSampler, Vocab, create_vocab
from trainer import Trainer
from model import Model
import torch
import numpy as np
import random
import datetime


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def train():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--dataset", type=str, help="train dataset", default='emse', choices=['funcom', 'emse', ])

    # dataset size
    parser.add_argument("--max_target_len", type=int, default=25, help=' <eos or sos> + true len of method name')
    parser.add_argument("--max_source_len", type=int, default=512, help=' <eos or sos> + true len of method name')
    # parser.add_argument("--infer_target_len", type=int, default=25, help=' <eos or sos> + true len of method name')
    # vocab
    parser.add_argument("--weight_tying", type=boolean_string, default=True,
                        help="right embedding = pre softmax matrix ")
    parser.add_argument("--src_vocab_size", type=int, default=50200, help="if use uni vocab, and use vocab threshold")
    parser.add_argument("--tgt_vocab_size", type=int, default=30000, help="if use uni vocab, and use vocab threshold")
    # EMSE: src:50000 tgt:30000
    # funcom: src 35000 tgt 30000
    # 为什么论文里写的词表是这么设置的？funcom的语料明明更大一些，但为什么词表却小一些？
    # 代码仓库里边对这个地方没有标注，只有5W和3W的配置文件
    # 所以至少对于emse来说，这个配置是正确的

    # trainer
    parser.add_argument("--length_group", type=boolean_string, default=True, )
    parser.add_argument("--gpu_nums", type=float, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.03, help="")
    parser.add_argument("--with_cuda", type=boolean_string, default=True, help="training with CUDA: true or false")
    parser.add_argument("--clip_grad", type=float, default=5.0, help="0 is no clip")
    parser.add_argument("--batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("--accu_batch_size", type=int, default=64,
                        help="number of real batch_size per step, setup for save gpu memory")
    parser.add_argument("--infer_batch_size", type=int, default=50, help="number of batch_size of infer")
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--num_workers", type=int, default=16, help="dataloader worker size")
    parser.add_argument("--save", type=boolean_string, default=True, help="whether to save model checkpoint")
    parser.add_argument("--dropout", type=float, default=0.1, help="0.1 in transformer paper")
    parser.add_argument("--lr", type=float, default=2e-4, help="")
    parser.add_argument("--milestones", type=int, default=28, help="")
    # [16, 24, 28] [16, 24, 32]
    parser.add_argument("--early_stop", type=int, default=8, help="")
    parser.add_argument("--warmup_steps", type=int, default=1500, help="")
    parser.add_argument("--warmup", type=boolean_string, default=True, )

    # transformer
    parser.add_argument("--embedding_size", type=int, default=512, help="hidden size of transformer model")
    parser.add_argument("--activation", type=str, default='relu', help="", choices=['gelu', 'relu'])
    parser.add_argument("--hidden", type=int, default=512, help="hidden size of transformer model")
    parser.add_argument("--d_ff_fold", type=int, default=4, help="ff_hidden = ff_fold * hidden")
    parser.add_argument("--e_ff_fold", type=int, default=4, help="ff_hidden = ff_fold * hidden")
    parser.add_argument("--encoder_layers", type=int, default=6, help="number of layers")
    parser.add_argument("--decoder_layers", type=int, default=6, help="number of decoder layers")
    parser.add_argument("--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("--xavier", type=boolean_string, default=True, help="")
    parser.add_argument("--pre_norm", type=boolean_string, default=True)
    parser.add_argument("--decoder_pre_norm", type=boolean_string, default=True)

    # Tree PE
    parser.add_argument("--max_depth", type=int, default=32, help="")
    parser.add_argument("--max_ary", type=int, default=64, help="")
    # parser.add_argument("--tree_PE", type=boolean_string, default=True, help="")
    parser.add_argument("--action_size", type=int, default=16, help="")

    # # Importance should change
    # parser.add_argument("--sqrt", type=int, default=2, help="")

    # Leaf PE
    # parser.add_argument("--leaf_PE", type=boolean_string, default=True, help="")
    parser.add_argument("--leaf_PE_Type", type=str, default='Merge', choices=['Merge', 'T5', 'Untied'], help="")
    parser.add_argument("--num_buckets", type=int, default=32, help="")
    parser.add_argument("--rot", type=boolean_string, default=False, )

    # Sentinel
    parser.add_argument("--type", type=str, default='sentinel', choices=['sentinel', 'tree', 'add', 'none', 'leaf'],
                        help="")
    parser.add_argument("--old_PE", type=boolean_string, default=False, help="")
    parser.add_argument("--sentinel_on_key", type=boolean_string, default=False, help="")
    parser.add_argument("--new_query", type=boolean_string, default=False, help="")

    # others
    parser.add_argument("--embedding_mul", type=boolean_string, default=True, help="")
    parser.add_argument("--pointer", type=boolean_string, default=True, help="")
    parser.add_argument("--pointer_leaf_mask", type=boolean_string, default=True, help="")

    # debug
    parser.add_argument("--seed", type=boolean_string, default=True, help="fix seed or not")
    parser.add_argument("--seed_idx", type=int, default=0, help="fix seed or not")
    parser.add_argument("--data_debug", type=boolean_string, default=False, help="check over-fit for a litter data")
    parser.add_argument("--train", type=boolean_string, default=True, help="Whether to train or just infer")
    # parser.add_argument("--shuffle", type=boolean_string, default=True, help="")
    parser.add_argument("--test_infer", type=boolean_string, default=True, help="")
    parser.add_argument("--tiny_data", type=int, default=0, help="only a little data for debug")
    parser.add_argument("--load_checkpoint", type=boolean_string, default=False,
                        help="load checkpoint for continue train or infer")
    parser.add_argument("--checkpoint", type=str, default='', help="load checkpoint for continue train or infer")
    parser.add_argument("--print_avg_gate", type=boolean_string, default=False, help="")
    # beam search
    parser.add_argument("--beam_search", type=boolean_string, default=True, help="")
    parser.add_argument("--width", type=int, default=3)

    args = parser.parse_args()
    print(args)
    args.batch_size = int(args.gpu_nums * args.batch_size)
    args.infer_batch_size = int(args.gpu_nums * args.infer_batch_size)
    if args.beam_search:
        args.infer_batch_size = args.infer_batch_size // args.width
    if args.seed:
        setup_seed(args.seed_idx)
    print('Experiment on {} dataset'.format(args.dataset))
    writer_path = '{}_{}'.format(args.dataset, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    print(writer_path)

    s_vocab = create_vocab(args.dataset, only_train=False, end='source', vocab_size=args.src_vocab_size,
                           no_special_token=True)
    t_vocab = create_vocab(args.dataset, only_train=False, end='target', vocab_size=args.tgt_vocab_size,
                           no_special_token=False)

    print("Loading Train Dataset")
    if args.data_debug:
        train_dataset = SCDataset(args, s_vocab, t_vocab, split='test')
        train_sampler = LengthGroupSampler(args, split='test', batch_size=args.accu_batch_size)
    else:
        train_dataset = SCDataset(args, s_vocab, t_vocab, split='train')
        train_sampler = LengthGroupSampler(args, split='train', batch_size=args.accu_batch_size)

    print("Loading Test Dataset")
    test_dataset = SCDataset(args, s_vocab, t_vocab, split='test')

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, sampler=train_sampler, collate_fn=SCDataset.collect_fn,
                                   batch_size=args.batch_size, num_workers=args.num_workers)
    test_infer_sampler = LengthGroupSampler(args, split='test', batch_size=args.infer_batch_size, mode='test')
    test_infer_data_loader = DataLoader(test_dataset, sampler=test_infer_sampler, collate_fn=SCDataset.collect_fn,
                                        batch_size=args.infer_batch_size, num_workers=args.num_workers, )
    print("Building Model")
    model = Model(args, s_vocab, t_vocab)

    print("Creating Trainer")
    trainer = Trainer(args=args, model=model, train_data=train_data_loader, test_infer_data=test_infer_data_loader,
                      t_vocab=t_vocab, writer_path=writer_path)

    if args.load_checkpoint:
        checkpoint_path = 'checkpoint/{}'.format(args.checkpoint)
        trainer.load(checkpoint_path)

    if not args.train:
        args.epochs = 1
    print("Training Start")
    for epoch in range(args.epochs):
        if args.train:
            trainer.train(epoch)
            trainer.scheduler_step()
        if args.test_infer:
            trainer.predict(epoch)
            if trainer.early_stop():
                break
    trainer.writer.close()


if __name__ == '__main__':
    train()
