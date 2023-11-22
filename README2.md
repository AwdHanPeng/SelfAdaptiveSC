现在正在做的实验：
1）CUDA_VISIBLE_DEVICES=0 python main.py --dataset=emse --type=sentinel --rot=True
emse_2023-01-12-14-42-40
###Best Result At EP27, best_bleu=42.716128837014146
{'bleu': 42.716128837014146, 'rouge_l': 58.224521508984495, 'meteor': 0, 'precision': 63.06560043875179, 'recall':
59.56787324502585, 'f1': 59.902424090662784}
done

2）CUDA_VISIBLE_DEVICES=3 python main.py --dataset=emse --type=none --decoder_pre_norm=False
emse_2023-01-12-14-37-36
###Best Result At EP28, best_bleu=37.2519073119773
{'bleu': 37.2519073119773, 'rouge_l': 53.06013746462703, 'meteor': 0, 'precision': 58.16559647131506, 'recall':
54.455605107269314, 'f1': 54.82879625549906}
done

3）CUDA_VISIBLE_DEVICES=7 python main.py --dataset=emse --type=sentinel --leaf_PE_Type=T5
emse_2023-01-12-13-17-52
###Best Result At EP29, best_bleu=42.425310448843184
{'bleu': 42.425310448843184, 'rouge_l': 57.92154942371219, 'meteor': 0, 'precision': 62.63269938315085, 'recall':
59.23869669931394, 'f1': 59.52554942444883}
done

4)CUDA_VISIBLE_DEVICES=2 python main.py --dataset=emse --type=tree
emse_2023-01-12-10-04-55
###Best Result At EP27, best_bleu=42.26482281364132
{'bleu': 42.26482281364132, 'rouge_l': 57.80424579809662, 'meteor': 0, 'precision': 62.54520350020149, 'recall':
59.20501549317898, 'f1': 59.472408427958044}
done

5)CUDA_VISIBLE_DEVICES=6 python main.py --dataset=emse --type=none
###Best Result At EP26, best_bleu=40.436023904502804
{'bleu': 40.436023904502804, 'rouge_l': 55.62439075394454, 'meteor': 0, 'precision': 60.24420895572764, 'recall':
57.096341464664526, 'f1': 57.31790621307778}
done

目前已有的结论：
1）decoder_pre_norm是导致结果和论文结果相差较大的原因，去除之后有明显的降低
2）sentinel相比于tree是有提升的，但是提升不大

目前已有的推测：
1）rot的相比于T5的sentinel应该是有一些改进的

现在需要补充的实验：
1）decoder_pre_norm场景下的tree模型，消融实验使用
CUDA_VISIBLE_DEVICES=2 python main.py --dataset=emse --type=tree --decoder_pre_norm=False
emse_2023-01-13-04-32-54
###Best Result At EP27, best_bleu=40.506187680831374
{'bleu': 40.506187680831374, 'rouge_l': 56.429181549155, 'meteor': 0, 'precision': 61.71565195059864, 'recall':
57.5951603646473, 'f1': 58.137277660199096}
done

2）decoder_pre_norm场景下的rot模型，作为最终结果
CUDA_VISIBLE_DEVICES=4 python main.py --dataset=emse --type=sentinel --rot=True --decoder_pre_norm=False
emse_2023-01-13-04-34-29
###Best Result At EP29, best_bleu=40.97499869858314
{'bleu': 40.97499869858314, 'rouge_l': 56.941402078655514, 'meteor': 0, 'precision': 62.014464732816386, 'recall':
58.20372894023234, 'f1': 58.61928546080465}
done

3）decoder_pre_norm场景下sentinelT5模型，作为和最终结果比较的结果
CUDA_VISIBLE_DEVICES=7 python main.py --dataset=emse --type=sentinel --leaf_PE_Type=T5 --decoder_pre_norm=False
emse_2023-01-13-04-36-40
###Best Result At EP27, best_bleu=41.16940393301995
{'bleu': 41.16940393301995, 'rouge_l': 57.046171027734495, 'meteor': 0, 'precision': 62.21428791592382, 'recall':
58.22710644020048, 'f1': 58.756986900874416}
done

一小时后之后将有一批结果会出，可以用来上新的思考，我的建议是不用思考太多，如果实在没有思路，目前的情况就已经可以了。
##############
结果挺好，t5的效果相比而言还要更高一些，
这样比较舒服，因为t5本身就比较简单。
然后消融实验部分，提升了0.6个点，挺不错的
所以现在这个结果已经就能用了

如果t5高的话，我觉得可以补上之前直接add的方式作为baseline写进去。
然后再补一个只在叶子节点上做leafPE的模型结果

#####

初步观察，发现在decoderprenorm的场景下
似乎t5的版本要比rot版本要好一些
########

想到一个改进的点，就是根据一篇文章的启发，准备把gate放在Query上，
然后设置一个h*dk的参数，然后让Query分别和这个参数计算，得出门控概率
开始测试
直接在decoderprenorm上进行实验，并且分别与t5和rot版本结合
CUDA_VISIBLE_DEVICES=0 python main.py --dataset=emse --type=sentinel --leaf_PE_Type=T5 --decoder_pre_norm=False
--sentinel_on_key=True
emse_2023-01-13-08-52-29
###Best Result At EP27, best_bleu=40.43359556937886
{'bleu': 40.43359556937886, 'rouge_l': 56.43840568911599, 'meteor': 0, 'precision': 61.64671227540674, 'recall':
57.624110575095564, 'f1': 58.13427253597162}
done
这个模型就基本上没有效果了已经，相比于之前的表现很差
####### 结论：只上一个512的参数，然后在query上直接计算效果不好

CUDA_VISIBLE_DEVICES=1 python main.py --dataset=emse --type=sentinel --rot=True --decoder_pre_norm=False
--sentinel_on_key=True
emse_2023-01-13-08-52-39
done
这个结果相比于对应的rot的之前的512*8的参数量的模型，明显是比不过的，所以可以直接cut掉

这样感觉确实更合理一些
这种的话 参数是512

其实还有另外一种可能性，就是新设置一个linear，content进去之后分头，然后再和哨兵计算
这样的话，参数量是512*512+512

之前最开始的是512*8

说实话我感觉参数量大的话，起作用的可能性会更高一点

结果还不清楚 只能等明天了

################
由于已经确定512参数量的模型效果不好，512*8的要更好一些，所以想进一步测试512*512+512参数量模型的表现。
现在还可以做的实验是
但是这样的话，512*512+512，由于每一层都需要做这样的操作，等于多了6倍的参数量，参数量是1.5m
512*8*6的话是，24k的参数
但是这个东西无论怎么样都必须得跟层数是相关的，必须得根据当前层的content进行计算才是合理的。
所以还是试一下吧。
1）
CUDA_VISIBLE_DEVICES=0 python main.py --dataset=emse --type=sentinel --leaf_PE_Type=T5 --decoder_pre_norm=False
--sentinel_on_key=True --new_query=True
emse_2023-01-14-15-24-41
###Best Result At EP25, best_bleu=39.57002162635927
{'bleu': 39.57002162635927, 'rouge_l': 55.69693260667766, 'meteor': 0, 'precision': 61.06279167520056, 'recall': 56.82024200311946, 'f1': 57.403761296603136}
done
基于postnorm，然后是哨兵版本，使用t5score，然后使用新的linear给context做变化，然后和哨兵进行计算
似乎相比起原来的，效果会变差
快跑完了，可以等跑完再cut也可以，不过基本确定这种设计没意义了。

2）
CUDA_VISIBLE_DEVICES=6 python main.py --dataset=emse --type=sentinel --rot=True --decoder_pre_norm=False
--sentinel_on_key=True --new_query=True
emse_2023-01-14-06-00-11
###Best Result At EP29, best_bleu=40.51328975381877
{'bleu': 40.51328975381877, 'rouge_l': 56.56433475454736, 'meteor': 0, 'precision': 61.822787558883775, 'recall':
57.70617892091051, 'f1': 58.247731437903724}
done
差了之前模型0.4个点，所以也是不work的。
基于postnorm，然后是哨兵版本，使用rot计算score，然后使用新的linear给context做变化，然后和哨兵进行计算

以上两个实验的目的是研究更多参数情况下是否还会有性能的进一步提升
---》结论：比较发现，基本上没有提升，且基本上都会有性能的损失，而且本身参数量还变大很多，所以不值得。

############
接下来的实验是只使用leafPE的结果，分为rot和T5两个版本的结果
3）CUDA_VISIBLE_DEVICES=7 python main.py --dataset=emse --type=leaf --rot=True --decoder_pre_norm=False
emse_2023-01-14-06-00-30
###Best Result At EP28, best_bleu=40.42624389342806
{'bleu': 40.42624389342806, 'rouge_l': 56.46019035676064, 'meteor': 0, 'precision': 61.56245144995416, 'recall':
57.699768074339595, 'f1': 58.130156579037696}
done
这个实验的目的是研究在仅使用leafPE时模型的效果是如何的，且基于rot以及postnorm

4）CUDA_VISIBLE_DEVICES=4 python main.py --dataset=emse --type=leaf --leaf_PE_Type=T5 --decoder_pre_norm=False
emse_2023-01-14-06-00-52
###Best Result At EP26, best_bleu=39.97027131446913
{'bleu': 39.97027131446913, 'rouge_l': 55.86924299254783, 'meteor': 0, 'precision': 60.8969906308077, 'recall':
57.26194004366884, 'f1': 57.57564222469589}
done
这个实验的目的是研究在仅使用leafPE时模型的效果是如何的，且基于T5以及postnorm

T5和rot都是要做的，因为目前还没有完全确定最终模型的形态。

有个事情一定要注意，就是add的这一种类型，是没办法和rot结合的，因为他的bias是直接和QKmerge到一起了

#############
现在缺一个这样的实验，首先就是vanilla add的结果
这个结果补上之后我觉得基本上就差不多了，

然后我觉得缺的实验就是在另外一个数据集上了，也就是在funcom上的实验结果

1）CUDA_VISIBLE_DEVICES=6 python main.py --dataset=emse --type=add --leaf_PE_Type=T5 --decoder_pre_norm=False
emse_2023-01-15-04-02-17
###Best Result At EP29, best_bleu=40.44646159966944
{'bleu': 40.44646159966944, 'rouge_l': 56.46810144417971, 'meteor': 0, 'precision': 61.655888054951355, 'recall': 57.73967508634801, 'f1': 58.206025750788136}
done

###################
然后就是funcom的实验
--src_vocab_size=35200 --tgt_vocab_size=30000 --epochs=36 --milestones=32 --max_target_len=35
CUDA_VISIBLE_DEVICES=1 python main.py --dataset=funcom --type=none --decoder_pre_norm=False --src_vocab_size=35200 --tgt_vocab_size=30000 --epochs=36 --milestones=32 --max_target_len=35
funcom_2023-01-15-04-36-53
done
###Best Result At EP34, best_bleu=42.56346811443474
{'bleu': 42.56346811443474, 'rouge_l': 58.2940651956048, 'meteor': 0, 'precision': 63.35823079434264, 'recall': 59.84178937712541, 'f1': 60.26745958487238}



CUDA_VISIBLE_DEVICES=5 python main.py --dataset=funcom --type=tree --decoder_pre_norm=False --src_vocab_size=35200 --tgt_vocab_size=30000 --epochs=36 --milestones=32 --max_target_len=35
funcom_2023-01-15-04-40-06
done
###Best Result At EP34, best_bleu=44.64115984739485
{'bleu': 44.64115984739485, 'rouge_l': 60.53947983057172, 'meteor': 0, 'precision': 66.12277522443625, 'recall': 61.9028328102459, 'f1': 62.620672605215624}


CUDA_VISIBLE_DEVICES=7 python main.py --dataset=funcom --type=sentinel --leaf_PE_Type=T5 --decoder_pre_norm=False --src_vocab_size=35200 --tgt_vocab_size=30000 --epochs=36 --milestones=32 --max_target_len=35
funcom_2023-01-15-04-41-49
done
###Best Result At EP32, best_bleu=44.252639629478956
{'bleu': 44.252639629478956, 'rouge_l': 60.18163813635506, 'meteor': 0, 'precision': 65.66511240941104, 'recall': 61.64821861085799, 'f1': 62.270809408510985}


CUDA_VISIBLE_DEVICES=0 python main.py --dataset=funcom --type=leaf --leaf_PE_Type=T5 --decoder_pre_norm=False --src_vocab_size=35200 --tgt_vocab_size=30000 --epochs=36 --milestones=32 --max_target_len=35
funcom_2023-01-15-04-43-40
done
###Best Result At EP34, best_bleu=44.02080411591911
{'bleu': 44.02080411591911, 'rouge_l': 59.84194799550985, 'meteor': 0, 'precision': 65.27817072208164, 'recall': 61.339289436282215, 'f1': 61.919151989736775}


CUDA_VISIBLE_DEVICES=3 python main.py --dataset=funcom --type=add --leaf_PE_Type=T5 --decoder_pre_norm=False --src_vocab_size=35200 --tgt_vocab_size=30000 --epochs=36 --milestones=32 --max_target_len=35
funcom_2023-01-15-10-30-27
done
###Best Result At EP34, best_bleu=44.445406912077615
{'bleu': 44.445406912077615, 'rouge_l': 60.37721577987929, 'meteor': 0, 'precision': 65.9012697408066, 'recall': 61.76956050648156, 'f1': 62.46557196058019}


add明显不如tree，这个和emse上的结论一致
但是为什么sentinel也弱于tree呢？
sentinel和leaf比怎么样？sentinel是要比leaf强的，
那我怎么感觉是leaf太弱的原因？
sentinel和add比呢？sentinel比不过add，这个情况有点不太合理
我感觉会不会是sentinel搞错了啊，检查了一下，感觉没发现啥问题。
那就再等等之后的结果吧，不行就不做消融实验了。

######

其实不用统计的，因为都是padding就可以了
#########
然后再把另外一个评估指标测一下就完事了


funcom上消融实验不是很乐观，似乎没起到什么作用，明天再看下结果，
如果不行的话，就得考虑再加个别的实验了。

看了一下，baseline也只是在emse上做了消融实验，
因为确实这个数据集上的结果咬的很紧。


#############
现在的目标是毕业为最高目标，发论文不是我需要考虑的事情，
所以以目前的情况来看，现在唯一的问题就只有一个：
就是我的模型的表现在funcom的数据集上的效果似乎只有44.3左右，
看这个样子是比不过paformer了，
为了解决这个问题，我觉得可以有这么几个策略：

第一个方案是，什么实验都不再跑了，也不要浪费时间调参了，没必要，
然后emse和funcom的实验都放进去，那么如果这样的话，
由于funcom没比过baseline，而且在funcom上所有的指标都超不过baseline，
所以只能不将paformer当作baseline了，
那么这种方案会有几个问题，
第一，发论文是百分之一百没办法发了，因为你现在用的结果都是paformer上截取的，
但是你却不和人家比，这算什么事？
第二，scformer在emse上的结果超过baseline可能有点多了，整整4个点，这个确实有点多，
多本身不是问题，问题是你的消融实验相比较而言就有点怪了，几个消融实验看下来，都比tptrans的37.24高了太多了。
不过这个方案也是可选的，毕竟毕业论文优先，其他都无所谓了。

说实话，这个方案目前看上去是最靠谱的
就是不把paformer当作baseline来比较


第二个方案，只用emse上的结果，不用funcom了，然后放上paformer作为baseline，
这样的话，实验部分就有点单薄了，只有一个任务一个实验，感觉不是很solid的样子，
不过好处是在emse数据集上表现是很舒服的，

第三个方案是，用prenorm跑一把，然后这样的后果就是emse和funcom上所有的结果都能狠狠的上去一大截，
但是这样的话，在这些数据集上的消融实验就会非常的扯淡，none版本的模型都要比大多数baseline要高，
这是非常不和逻辑的，而且还有一个大问题是需要重新跑，也就意味着t5这些模型的超参数选择也都不确定了，
没有办法搞了，。这样唯一带来的好处就是拿到了一个假的sota，而且这个sota本质上还是作弊得来的结果，
不是很好其实。所以这个选择综合来看还是不考虑。

现在思考一下是否可不可以这样？在emse上用postnorm，然后在funcom上用prenorm？
因为在emse上用prenorm消融实验实在是不过关，而且会特别的高，没办法说明问题。
在funcom上如果用prenorm的话，似乎能和paformer的效果相持平，但是效果也估计没办法超出太多的样子。

要是在funcom上用postnorm然后效果能再高一点点就好了，一切都好解决了，




第四个方案是，只用emse上的结果，带着paformer，然后呢，再加一个emnlp文章的数据集跑一把实验，
也就是再加一把方法名的实验跑一波，这样的好处是，
其一，实验更多样化，等于有了两个不同的任务，
其二，方法名如果跑的好的话，是可以在两个数据集上都有效果的，所以也就有了两个语言，
其三，baseline部分引入了emnlp文章的结果，对于毕业论文来说是既为有利的
这个最终证实的结果是再emnlp数据集上效果极差，在td的基础上加入gate确实有一些提升，
但是tdlr加到一起，然后再放入gate，效果反而非常的差，感觉这个数据集给我的感觉就是很烂的一个数据集。



###########
看了funcom确实没法用，必须得早点行动。

###############
现在js上的fn也毙掉了 只剩下看postnorm版本怎么样了。
CUDA_VISIBLE_DEVICES=0 python main.py --dataset=funcom --type=sentinel --leaf_PE_Type=T5 --decoder_pre_norm=True --src_vocab_size=35200 --tgt_vocab_size=30000 --epochs=36 --milestones=32 --max_target_len=35
funcom_2023-01-18-16-42-28
###Best Result At EP34, best_bleu=44.876400340776065
{'bleu': 44.876400340776065, 'rouge_l': 60.61856626618234, 'meteor': 0, 'precision': 65.84902669388926, 'recall': 62.12737110141919, 'f1': 62.625658850317876}
done

CUDA_VISIBLE_DEVICES=5 python main.py --dataset=funcom --type=tree --decoder_pre_norm=True --src_vocab_size=35200 --tgt_vocab_size=30000 --epochs=36 --milestones=32 --max_target_len=35
funcom_2023-01-18-16-43-28
###Best Result At EP32, best_bleu=44.75772730687724
{'bleu': 44.75772730687724, 'rouge_l': 60.618149362816084, 'meteor': 0, 'precision': 65.84473222008363, 'recall': 62.11670702722192, 'f1': 62.6132238153428}
done

在这个数据集上，加了postnorm好像效果也没有压倒性优势，
bleu值稍微高一些，然后其他指标都超不过
而且消融实验也是没有什么明显的提升。




算了，不想搞了，明天看看加上tdlr以及sentinel的效果如何，
如果是在不行，那就再想想别的解决方案吧

那如果是这样的话，那就只好上prenorm了，然后这样把funcom也冲到sota就完事。

看了一下，似乎在funcom上prenorm和postnorm到后期相差没有那么的大了。
而且tree和sentinel之间相差同样也没有那么的大。
所以目前的状况感觉只有这样一个解决方案了：
就是不把paformer当作baseline，然后emse和funcom的实验都放上去，
整理一下，看看是多少。
然后如果要发论文的话，那就只好之后再说了。


或者换个种子在sentinel以及prenorm=False上跑一下
CUDA_VISIBLE_DEVICES=3 python main.py --seed_idx=1 --dataset=funcom --type=sentinel --leaf_PE_Type=T5 --decoder_pre_norm=False --src_vocab_size=35200 --tgt_vocab_size=30000 --epochs=36 --milestones=32 --max_target_len=35
funcom_2023-01-19-09-16-31
###Best Result At EP30, best_bleu=44.54407749437988
{'bleu': 44.54407749437988, 'rouge_l': 60.42895099744404, 'meteor': 0, 'precision': 65.91131462950028, 'recall': 61.78849410446021, 'f1': 62.48414881583636}
done

CUDA_VISIBLE_DEVICES=4 python main.py --seed_idx=2 --dataset=funcom --type=sentinel --leaf_PE_Type=T5 --decoder_pre_norm=False --src_vocab_size=35200 --tgt_vocab_size=30000 --epochs=36 --milestones=32 --max_target_len=35
funcom_2023-01-19-09-16-56
###Best Result At EP33, best_bleu=44.68303690251153
{'bleu': 44.68303690251153, 'rouge_l': 60.518408430762705, 'meteor': 0, 'precision': 65.99007301411956, 'recall': 61.91679149006662, 'f1': 62.58632785693562}
done


选这个结果

CUDA_VISIBLE_DEVICES=6 python main.py --seed_idx=3 --dataset=funcom --type=sentinel --leaf_PE_Type=T5 --decoder_pre_norm=False --src_vocab_size=35200 --tgt_vocab_size=30000 --epochs=36 --milestones=32 --max_target_len=35
funcom_2023-01-19-09-17-05
###Best Result At EP35, best_bleu=44.55437598649525
{'bleu': 44.55437598649525, 'rouge_l': 60.421870111052144, 'meteor': 0, 'precision': 66.01121137271014, 'recall': 61.7784633269784, 'f1': 62.50635013847165}
done

这个结果说明，换了种子之后效果一般会好一些，说明原来的种子确实比较差，
那我觉得可以选最好的那个也可以。


有没有这样一种可能，把funcom的词表从35000调大到50000？
也就是和emse持平，然后在decoderprenorm=False上跑一波？
因为我确实有点怀疑，为什么数据集大反而词表却设的小？
CUDA_VISIBLE_DEVICES=1 python main.py --dataset=funcom --type=sentinel --leaf_PE_Type=T5 --decoder_pre_norm=False --src_vocab_size=50200 --tgt_vocab_size=30000 --epochs=36 --milestones=32 --max_target_len=35
funcom_2023-01-20-05-44-21
###Best Result At EP27, best_bleu=44.510348259658805
{'bleu': 44.510348259658805, 'rouge_l': 60.40701493516529, 'meteor': 0, 'precision': 65.88763243850798, 'recall': 61.812234053147044, 'f1': 62.481450885030064}
done
增大词表之后有约为0.2个点的提升
但是和paformer相比还是有一些差距在的。
所以综合考虑的话，这个版本的结果还是不怎么想用的




其他的实验目前只能等了，感觉也没有进一步做的必要了。

现在看起来，实验没有任何的进步了，只能如此了。
只能说等一下换种子的funcom结果，然后在和弱baseline比的时候尽量让所有的指标都
超过baseline。
