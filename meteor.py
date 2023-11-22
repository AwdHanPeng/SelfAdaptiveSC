# from nltk.translate.meteor_score import single_meteor_score
# import nltk
# nltk.download('wordnet')
import os
import argparse
# import torch
# import evaluate
from evaluation.meteor.meteor import Meteor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="emse_2023-01-13-04-36-40")
    parser.add_argument('--epoch', type=int, default=27)
    args = parser.parse_args()
    print(args)
    # meteor = evaluate.load("meteor")
    dir = args.dir
    epoch = args.epoch
    ref_file = os.path.join('./run', dir, 'ref_test.txt')
    pred_file = os.path.join('./run', dir, 'pred_test_{}.txt'.format(epoch))
    ref_list = []
    result = []
    # with open(ref_file, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         ref_list.append(line.split())
    # pred_list = []
    # with open(pred_file, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         pred_list.append(line.split())
    # for ref, pred in zip(ref_list, pred_list):
    #     # result.append(single_meteor_score(ref, pred))
    #
    # print(torch.tensor(result).mean().item())

    # with open(ref_file, 'r') as f:
    #     ref_lines = f.readlines()
    # with open(pred_file, 'r') as f:
    #     pred_lines = f.readlines()
    # print(meteor.compute(predictions=pred_lines, references=ref_lines))
    references, hypotheses = dict(), dict()
    ref_lines, pred_lines = [], []
    with open(ref_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            ref_lines.append(line.strip())
    with open(pred_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            pred_lines.append(line.strip())
    # print(pred_lines)
    for idx, (ref, pred) in enumerate(zip(ref_lines, pred_lines)):
        # print(ref, pred)
        references[idx] = [ref]
        hypotheses[idx] = [pred]

    meteor_calculator = Meteor()
    meteor, _ = meteor_calculator.compute_score(references, hypotheses)
    print(meteor)
    # emse
    # emse_2023-01-13-04-36-40  27 sentinel  0.270973183748721
    # emse_2023-01-13-04-32-54 27 tree 0.267122095440045
    # emse_2023-01-14-06-00-52 26 leaf 0.2641142170385753
    # emse_2023-01-12-14-37-36 28 none 0.24349465780763616
    # emse_2023-01-15-04-02-17 29 add 0.2674744439112246

    # funcom
    # funcom_2023-01-15-04-41-49 32 sentinel 0.28106153783364857

    # funcom_2023-01-19-09-16-56 33 sentinel 0.283407025737665