from tree_sitter import Language, Parser
import json
import os
from tqdm import tqdm
import jsonlines
import joblib
from typing import List, Dict


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def init_parser(language):
    Language.build_library(
        './build/{}.so'.format(language),
        [
            './vendor/tree-sitter-{}'.format(language),
        ]
    )
    language = Language('./build/{}.so'.format(language), language)
    lang_parser = Parser()
    lang_parser.set_language(language)
    return lang_parser


def read_files(dataset, split):
    '''
    sub_token_seq: ["protected", "vpn", "daemons", "get", "daemons", "(", ")", "{", "return", "m", "daemons", ";", "}"]
    token_seq: ["protected", "VpnDaemons", "getDaemons", "(", ")", "{", "return", "mDaemons", ";", "}"]
    summary_seq: ["Returns", "the", "daemons", "management", "class", "for", "this", "service", "object", "."]
    code: "public class A09360549 {\n     protected VpnDaemons getDaemons() {\n        return mDaemons;\n    }\n  \n}"
    :param dataset:
    :param split:
    :return:
    '''
    file_path = '{}_{}.jsonl'.format(dataset, split)
    all_data = []
    with jsonlines.open(os.path.join("../raw_data", file_path), 'r') as reader:
        for sample in reader:
            all_data.append({'sub_token_seq': sample['sub_token_seq'],
                             'token_seq': sample['token_seq'],
                             "summary_seq": sample['summary_seq'],
                             "code": sample["code"],
                             "sub_token_to_token": sample["sub_token_to_token"]})
    print('Load {} {} files => {}'.format(dataset, split, len(all_data)))
    return all_data


def token_statistic(source_dic: Dict, target_dic: Dict, source: List, target: List):
    def lookup_update(dic, item):
        if item in dic:
            dic[item] += 1
        else:
            dic[item] = 1
        return list(dic.keys()).index(item)

    for token in source:
        _ = lookup_update(source_dic, token)
    for token in target:
        _ = lookup_update(target_dic, token)


def update_sum_dict(sub_source_dic, source_dic, sub_target_dic, target_dic):
    def update_dict(sub_dic, dic):
        for key, value in sub_dic.items():
            if key in dic:
                dic[key] += value
            else:
                dic[key] = value

    update_dict(sub_source_dic, source_dic)
    update_dict(sub_target_dic, target_dic)
