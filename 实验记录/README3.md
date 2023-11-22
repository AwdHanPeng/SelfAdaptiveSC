CUDA_VISIBLE_DEVICES=3 python main.py --dataset=emse --type=none --decoder_pre_norm=False
emse_2023-01-12-14-37-36
###Best Result At EP28, best_bleu=37.2519073119773
{'bleu': 37.2519073119773, 'rouge_l': 53.06013746462703, 'meteor': 0, 'precision': 58.16559647131506, 'recall':
54.455605107269314, 'f1': 54.82879625549906}

CUDA_VISIBLE_DEVICES=2 python main.py --dataset=emse --type=tree --decoder_pre_norm=False
emse_2023-01-13-04-32-54
###Best Result At EP27, best_bleu=40.506187680831374
{'bleu': 40.506187680831374, 'rouge_l': 56.429181549155, 'meteor': 0, 'precision': 61.71565195059864, 'recall':
57.5951603646473, 'f1': 58.137277660199096}

CUDA_VISIBLE_DEVICES=4 python main.py --dataset=emse --type=sentinel --rot=True --decoder_pre_norm=False
emse_2023-01-13-04-34-29
###Best Result At EP29, best_bleu=40.97499869858314
{'bleu': 40.97499869858314, 'rouge_l': 56.941402078655514, 'meteor': 0, 'precision': 62.014464732816386, 'recall':
58.20372894023234, 'f1': 58.61928546080465}

CUDA_VISIBLE_DEVICES=7 python main.py --dataset=emse --type=sentinel --leaf_PE_Type=T5 --decoder_pre_norm=False
emse_2023-01-13-04-36-40
###Best Result At EP27, best_bleu=41.16940393301995
{'bleu': 41.16940393301995, 'rouge_l': 57.046171027734495, 'meteor': 0, 'precision': 62.21428791592382, 'recall':
58.22710644020048, 'f1': 58.756986900874416}

CUDA_VISIBLE_DEVICES=7 python main.py --dataset=emse --type=leaf --rot=True --decoder_pre_norm=False
emse_2023-01-14-06-00-30
###Best Result At EP28, best_bleu=40.42624389342806
{'bleu': 40.42624389342806, 'rouge_l': 56.46019035676064, 'meteor': 0, 'precision': 61.56245144995416, 'recall':
57.699768074339595, 'f1': 58.130156579037696}

CUDA_VISIBLE_DEVICES=4 python main.py --dataset=emse --type=leaf --leaf_PE_Type=T5 --decoder_pre_norm=False
emse_2023-01-14-06-00-52
###Best Result At EP26, best_bleu=39.97027131446913
{'bleu': 39.97027131446913, 'rouge_l': 55.86924299254783, 'meteor': 0, 'precision': 60.8969906308077, 'recall':
57.26194004366884, 'f1': 57.57564222469589}

CUDA_VISIBLE_DEVICES=6 python main.py --dataset=emse --type=add --leaf_PE_Type=T5 --decoder_pre_norm=False
emse_2023-01-15-04-02-17
###Best Result At EP29, best_bleu=40.44646159966944
{'bleu': 40.44646159966944, 'rouge_l': 56.46810144417971, 'meteor': 0, 'precision': 61.655888054951355, 'recall':
57.73967508634801, 'f1': 58.206025750788136}
done

这个结果基本可以确定了，就选择T5版本作为最终模型结果