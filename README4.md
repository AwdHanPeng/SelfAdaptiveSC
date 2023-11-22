这个文件整理了所有prenorm=True的模型结果。

CUDA_VISIBLE_DEVICES=0 python main.py --dataset=funcom --type=sentinel --leaf_PE_Type=T5 --decoder_pre_norm=True --src_vocab_size=35200 --tgt_vocab_size=30000 --epochs=36 --milestones=32 --max_target_len=35
funcom_2023-01-18-16-42-28
doing

CUDA_VISIBLE_DEVICES=5 python main.py --dataset=funcom --type=tree --decoder_pre_norm=True --src_vocab_size=35200 --tgt_vocab_size=30000 --epochs=36 --milestones=32 --max_target_len=35
funcom_2023-01-18-16-43-28
doing

整体比对应的高0.6个点，sentinel比tree高0.1个点。感觉不大行的样子。

############################

CUDA_VISIBLE_DEVICES=7 python main.py --dataset=emse --type=sentinel --leaf_PE_Type=T5
emse_2023-01-12-13-17-52
###Best Result At EP29, best_bleu=42.425310448843184
{'bleu': 42.425310448843184, 'rouge_l': 57.92154942371219, 'meteor': 0, 'precision': 62.63269938315085, 'recall':
59.23869669931394, 'f1': 59.52554942444883}

CUDA_VISIBLE_DEVICES=1 python main.py --dataset=emse --type=leaf --leaf_PE_Type=T5 --decoder_pre_norm=True
emse_2023-01-19-06-17-18
###Best Result At EP27, best_bleu=41.29208389474916
{'bleu': 41.29208389474916, 'rouge_l': 56.48627457573633, 'meteor': 0, 'precision': 61.13386534185031, 'recall': 57.91171970714096, 'f1': 58.14987569471812}
done

CUDA_VISIBLE_DEVICES=2 python main.py --dataset=emse --type=tree
emse_2023-01-12-10-04-55
###Best Result At EP27, best_bleu=42.26482281364132
{'bleu': 42.26482281364132, 'rouge_l': 57.80424579809662, 'meteor': 0, 'precision': 62.54520350020149, 'recall':
59.20501549317898, 'f1': 59.472408427958044}

CUDA_VISIBLE_DEVICES=6 python main.py --dataset=emse --type=none
###Best Result At EP26, best_bleu=40.436023904502804
{'bleu': 40.436023904502804, 'rouge_l': 55.62439075394454, 'meteor': 0, 'precision': 60.24420895572764, 'recall':
57.096341464664526, 'f1': 57.31790621307778}

这样整不行啊，消融实验两个结果太像了，不过关啊。