import argparse
from utils import init_parser, read_files, boolean_string, token_statistic, update_sum_dict
import os
import random
from multiprocessing import Process
from tqdm import tqdm
# import joblib
import json
import pickle as joblib


class TreeNode(object):
    def __init__(self, inter_node, token):
        self.child = []
        self.inter_node = inter_node
        self.token = token

    def add_child(self, node):
        if node not in self.child:
            self.child.append(node)


def language_parse(args, data, lang_parser):
    code = data['code']
    sub_token_seq = data['sub_token_seq']
    token_seq = data['token_seq']
    sub_token_to_token = data['sub_token_to_token']  # use this to rewrite the preorder func
    tree = lang_parser.parse(bytes(code, "utf-8"))
    tree_nodes = []
    node_actions = []
    father_actions = []

    root = None
    token_idx = 0

    def get_func_node(root):
        temp = [root]
        while len(temp) != 0:
            for _ in range(len(temp)):
                cur = temp.pop(0)
                if cur.type in ['method_declaration', 'constructor_declaration']:
                    return cur
                else:
                    for child in cur.children:
                        temp.append(child)
        return None

    def preorder(node):
        if node.child_count == 0:
            if 'comment' in node.type.lower() or 'doc' in node.type.lower():
                return

            node_text = node.text.decode()
            nonlocal token_idx

            while node_text != token_seq[token_idx]:  # 这个是用来跳过后期被裁切部分token的样本的，必须得有
                token_idx += 1

            if node_text != token_seq[token_idx]:
                print(token_seq)
                assert node_text == token_seq[token_idx]

            if sub_token_to_token.count(token_idx) == 0:
                token_idx += 1
                return

            sub_token_idx_start = sub_token_to_token.index(token_idx)
            sub_token_idx_end = sub_token_idx_start + sub_token_to_token.count(token_idx)  # [)

            token_idx += 1

            if node.type == node_text:

                assert sub_token_idx_end - sub_token_idx_start == 1  # for keyword in syntax node: protected ect

                if node_text != sub_token_seq[sub_token_idx_start].lower():
                    assert node_text == sub_token_seq[sub_token_idx_start].lower()

                father_node = TreeNode(inter_node=False, token=node.type)
                tree_nodes.append(father_node)
            else:
                father_node = TreeNode(inter_node=True, token=node.type)  # identifier node
                tree_nodes.append(father_node)
                for idx in range(sub_token_idx_start, sub_token_idx_end):
                    s = sub_token_seq[idx]
                    # if s not in node_text.lower():
                    #     assert s in node_text.lower()
                    child_node = TreeNode(inter_node=False, token=s)
                    tree_nodes.append(child_node)
                    father_node.add_child(child_node)
        else:
            father_node = TreeNode(inter_node=True, token=node.type)
            tree_nodes.append(father_node)
            for child in node.children:
                child_node = preorder(child)
                if child_node:
                    father_node.add_child(child_node)
        return father_node

    def preorder_action(node, action, father):
        node_actions.append(action)
        father_actions.append(father)
        count = 0
        total = len(node.child)
        for child in node.child:
            count += 1
            preorder_action(child, action + [count], father + [total])

    cursor = tree.walk()
    root = get_func_node(cursor.node)
    if root == None:
        # print(data)
        root = cursor.node
        # assert root is not None
        # 只要保证seq对上就可以了，保持一致就行了
    preorder(root)

    my_sub_token_seq = [node.token for node in tree_nodes if not node.inter_node]

    if sub_token_seq != my_sub_token_seq:
        assert sub_token_seq == my_sub_token_seq

    all_tokens = []
    leaf_idx = []

    count = 1
    for i, node in enumerate(tree_nodes):
        all_tokens.append(node.token)

        if node.inter_node:
            leaf_idx.append(0)
        else:
            leaf_idx.append(count)
            count += 1

    preorder_action(tree_nodes[0], [1], [1])

    assert len(node_actions) == len(tree_nodes)

    return all_tokens, leaf_idx, node_actions, father_actions


def sub_process(args, idx, all_data, lang_parser):
    save_path = os.path.join('../data', args.dataset, '{}_{}.pkl'.format(args.split, idx))
    data_list = []
    source_dic, target_dic = dict(), dict()
    source_dic_path = os.path.join('../data', args.dataset, '{}_{}_source_dic.json'.format(args.split, idx))
    target_dic_path = os.path.join('../data', args.dataset, '{}_{}_target_dic.json'.format(args.split, idx))
    # 现在主要存在这么一种情况，就是有些样本解析出来有问题，比方说 <<List>> 现在这个parser把右边两个>给莫名其妙粘起来了，就很蠢
    with open(save_path, 'wb') as f:
        for data in tqdm(all_data):
            try:
                all_tokens, leaf_idx, node_actions, father_actions = language_parse(args, data, lang_parser)
                data = {'target': [word.lower() for word in data['summary_seq']], 'input': all_tokens,
                        'node_actions': node_actions,
                        'leaf_idx': leaf_idx, 'father_actions': father_actions, }
                data_list.append(data)
                token_statistic(source_dic, target_dic, data['input'], data['target'])
            except Exception as e:
                print(e)
                print('Sample Can Not Been Parserd')
                print(data['code'])

        joblib.dump(data_list, f)
    with open(source_dic_path, 'w') as f_1:
        json.dump(source_dic, f_1)
    with open(target_dic_path, 'w') as f_2:
        json.dump(target_dic, f_2)


def process(args):
    lang_parser = init_parser('java')
    if not os.path.exists('../data/{}'.format(args.dataset)):
        os.makedirs('../data/{}'.format(args.dataset))
    all_data = read_files(args.dataset, args.split)
    if args.debug_idx >= 0: all_data = [all_data[args.debug_idx]]
    if args.shuffle: random.shuffle(all_data)
    if args.nums > 0: all_data = all_data[:args.nums]

    if args.process_num > 1:
        pool = []
        split_data = [[] for _ in range(args.process_num)]
        for i in range(len(all_data)):
            split_data[i % args.process_num].append(all_data[i])
        for i in range(args.process_num):
            pool.append(Process(target=sub_process, args=(args, i, split_data[i], lang_parser)))
            pool[-1].start()
        for p in pool:
            p.join()
    else:
        sub_process(args, 0, all_data, lang_parser)

    print('Sub Files Merge')

    sum_save_path = os.path.join('../data', args.dataset, '{}.pkl'.format(args.split))
    f_sum = open(sum_save_path, 'wb')
    length_list = []
    offset_list = []
    source_dic, target_dic = dict(), dict()

    for i in tqdm(range(args.process_num)):
        sub_save_path = os.path.join('../data', args.dataset, '{}_{}.pkl'.format(args.split, i))
        with open(sub_save_path, 'rb') as l:
            sub_data_list = joblib.load(l)
            for data in tqdm(sub_data_list):
                offset = f_sum.tell()
                offset_list.append(offset)
                length_list.append([len(data['input']), len(data['target'])])
                joblib.dump(data, f_sum)
        os.remove(sub_save_path)
        sub_source_dic_path = os.path.join('../data', args.dataset, '{}_{}_source_dic.json'.format(args.split, i))
        sub_target_dic_path = os.path.join('../data', args.dataset, '{}_{}_target_dic.json'.format(args.split, i))
        with open(sub_source_dic_path, 'r') as f_1:
            sub_source_dic = json.load(f_1)
        with open(sub_target_dic_path, 'r') as f_2:
            sub_target_dic = json.load(f_2)
        update_sum_dict(sub_source_dic, source_dic, sub_target_dic, target_dic)
        os.remove(sub_source_dic_path)
        os.remove(sub_target_dic_path)
    # data_list = []
    # for i in tqdm(range(args.process_num)):
    #     sub_save_path = os.path.join('../data', args.dataset, '{}_{}.pkl'.format(args.split, i))
    #     with open(sub_save_path, 'rb') as l:
    #         sub_data_list = joblib.load(l)
    #         data_list += sub_data_list
    #     os.remove(sub_save_path)

    # for data in tqdm(data_list):
    #     token_statistic(source_dic, target_dic, data['input'], data['target'])
    # joblib.dump(data_list, f_sum)

    # for data in tqdm(data_list):
    #     offset = f_sum.tell()
    #     offset_list.append(offset)
    #     length_list.append([len(data['input']), len(data['target'])])
    #     token_statistic(source_dic, target_dic, data['input'], data['target'])
    #     joblib.dump(data, f_sum)

    offset_path = os.path.join('../data', args.dataset, '{}_offset.pkl'.format(args.split))
    length_path = os.path.join('../data', args.dataset, '{}_lengths.pkl'.format(args.split))
    with open(offset_path, 'wb') as f:
        joblib.dump(offset_list, f)
    with open(length_path, 'wb') as f:
        joblib.dump(length_list, f)
    source_dic_path = os.path.join('../data', args.dataset, '{}_source_dic.json'.format(args.split))
    target_dic_path = os.path.join('../data', args.dataset, '{}_target_dic.json'.format(args.split))
    with open(source_dic_path, 'w') as f_1:
        json.dump(source_dic, f_1)
    print('Len(Source_Dic)=>{}'.format(len(source_dic)))
    with open(target_dic_path, 'w') as f_2:
        json.dump(target_dic, f_2)
    print('Len(Target_Dic)=>{}'.format(len(target_dic)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['funcom', 'emse'], type=str, default='emse')
    parser.add_argument('--split', choices=['train', 'test'], type=str, default='train')
    parser.add_argument('--process_num', type=int, default=1)
    parser.add_argument('--shuffle', type=boolean_string, default=False)
    parser.add_argument('--nums', type=int, default=-1)
    parser.add_argument('--debug_idx', type=int, default=-1)
    args = parser.parse_args()
    print(args)
    process(args)
