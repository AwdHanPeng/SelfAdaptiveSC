idea：
使模型能自适应的根据任务动态的调整structure和context的比重
方案：
输入所有的ast node到transformer里，然后所有的node之间做selfattn之后计算得到score，
随后我们分别计算tree上的PE score和context上的PE score，
让node的score分别和PE score分别组合以得到两个部分的概率分布，
然后在每一层我们使用一个哨兵向量，使用该向量向每一个node进行问询，判断每个节点更属于ast的一部分还是更属于context的一部分，
随后做sigmoid二分类之后，将每个node上的概率分布进行组合，
随后进行weight sum

#######################
`试验记录
`emse：
train 295967
train数据集有八个样本无法完美的解析成功
Len(Source_Dic)=>37544
Len(Target_Dic)=>49639

test 12226
其中test数据集解析完好
Len(Source_Dic)=>9485
Len(Target_Dic)=>8873

funcom：
train 1017964
train 里边11个样本无法完美的解析成功
Len(Source_Dic)=>220844
Len(Target_Dic)=>256950

test 53936
Len(Source_Dic)=>37316
Len(Target_Dic)=>35096
test数据集解析完好

#########
首先在funcom的一千条test数据集上进行debug实验

现在模型在非beam和非merge的场景下已经能顺利过拟合

beam 以及 sentinel似乎都能很好的过拟合，还未跑完

而且看到很明显的结果，sentinel相比于vanilla过拟合的速度快很多，bleu升的很快

################
2023/1/11
今天遇到一问题，发现预处理的特别慢，本来以为是dump随着文件的变大，往后再添加东西会降速
后来发现不是，因为理论上dump添加文件在句柄后边应该是linear的复杂度，不可能会降速的
所以后来经过排查，发现确实不是这个原因，而是我之前的这一个版本是将词表统计放到了主进程里
是词表统计变得慢了，导致时间变慢，这个具体原因我还不太清楚，因为理论上来说，词表这一块应该也是o1的复杂度啊
为啥会变慢呢

哦哦哦哦，那只有一个解释了，将词表统计下发到子进程等于加快了速度，合理，所以时间开销就完全只是这个原因出了问题，
教训就是，能下发进行多进程的任务，已经要下发，因为面对大数据情况，一点时间的丢失都会放大。

没有想到统计词表居然花费了这么长时间，感觉统计此表也占了大头

############
下午到三点前的任务，检查代码的正确性
随后在三点之前，观察跑完的debug是否有问题，比对门控概率进行初步的观察
随后设计几个不同的实验直接跑最终结果，
其中包括主实验，消融实验等

#########
现在有一些配置上的问题
结合论文和代码仓库中的设置可以发现

emse的配置是
训练30轮
src:50000 tgt:30000
lr_scheduler:
name: 'multi_step'
steps: [16, 24, 28]
gamma: 0.1
train:
start_epoch: 0
epochs: 30
early_stop: 8

而funcom的配置是
src 35000 tgt 30000
训练36轮
lr_scheduler:
name: 'multi_step'
steps: [16, 24, 32]
gamma: 0.1
train:
start_epoch: 0
epochs: 36
early_stop: 8

且目前的状况是emse的配置应该是完全正确的
funcom的设置有一些存疑
因为不知道为什么funcom的语料要相对大一些，词表却设置的小
但是不论怎么样，按照论文上来估计还是靠谱的
不放心可以问一下

#################
目前来看 已经没有什么问题了
可以直接跑实验了
且初步观察了一下过拟合实验，发现门控概率确实是在0.5左右徘徊的，有大有小

###########
目前需要做那些实验？
首先由于对leaf PE不是很确定，感觉这个部分可以考察一下
leaf_PE_Type = merge,T5,Untied

这个部分merge需要除以2吗？感觉不用就可以了，t5的那个基本可以忽略不计，我觉得没有必要对这个进行实验

CUDA_VISIBLE_DEVICES=0 python main.py --dataset=emse --leaf_PE_Type=Merge --type=sentinel
emse_2023-01-11-07-57-53
###Best Result At EP27, best_bleu=42.577096617000024

CUDA_VISIBLE_DEVICES=3 python main.py --dataset=emse --leaf_PE_Type=T5 --type=sentinel
emse_2023-01-11-08-00-47
###Best Result At EP29, best_bleu=42.93268187509514
{'bleu': 42.93268187509514, 'rouge_l': 58.35130542001874, 'meteor': 0, 'precision': 63.053100759716244, 'recall':
59.76692741367864, 'f1': 60.04576712369732}

CUDA_VISIBLE_DEVICES=1 python main.py --dataset=emse --leaf_PE_Type=Untied --type=sentinel
emse_2023-01-11-10-07-17
###Best Result At EP29, best_bleu=42.78264006848852

然后需要做的实验是消融实验：
第一个需要比的就是只要tree的实验结果，这个实验和leafPE选用哪种type是没有关系的
CUDA_VISIBLE_DEVICES=2 python main.py --dataset=emse --type=tree
emse_2023-01-11-10-06-48
这个实验很关键，和上边的对比一定要有提升才是合理的
###Best Result At EP29, best_bleu=42.62465022669503
{'bleu': 42.62465022669503, 'rouge_l': 58.1423088074463, 'meteor': 0, 'precision': 62.95598613807052, 'recall':
59.44372643445279, 'f1': 59.79364141112671}

然后再加几个别的对比实验
比如这个add，这个就是为了验证哨兵的作用
注意这个跑的时候需要和不同的leafType做对比
CUDA_VISIBLE_DEVICES=4 python main.py --dataset=emse --type=add --leaf_PE_Type=Merge
emse_2023-01-11-08-30-36
###Best Result At EP26, best_bleu=42.36603032968839

CUDA_VISIBLE_DEVICES=5 python main.py --dataset=emse --type=add --leaf_PE_Type=T5
emse_2023-01-11-08-29-54
###Best Result At EP28, best_bleu=42.4399894950381

CUDA_VISIBLE_DEVICES=6 python main.py --dataset=emse --type=add --leaf_PE_Type=Untied
emse_2023-01-11-08-30-14
###Best Result At EP26, best_bleu=42.324271241964915

CUDA_VISIBLE_DEVICES=7 python main.py --dataset=emse --type=none
emse_2023-01-11-14-29-49
###Best Result At EP29, best_bleu=40.62549158300682

############
现在遇到几个问题，第一个问题是，为什么我的结果这么高？
即使type=None，也就是没有任何树结构先验，模型的效果也这么好

第二个问题是，消融实验整体是合格的，但是差距没有拉开
只使用树模型和使用哨兵门控的模型两者之间差了不到0.5个bleu
不过按照论文里的结果来看，别的模型基本也都拉不开，基本都是0.2个bleu的提升，
所以只能说凑合吧

目前好的消息是，消融实验能看出来是合理的，
哨兵要比纯树模型要高，然后纯add模型始终没有哨兵模型好
我对现状的判断是，自然性建模还是差，就是针对叶子的位置编码这一块，

#############
现在解决两个问题，一个是想办法让哨兵版本的效果再提升一点
现在的观察给我的感觉可能有两个原因
一个是leafPE设计的还是不够好
第二个是哨兵还有进一步设计的可能吗？

第二个问题，结果和论文里边放的对不上
现在感觉有点差别的就是max target len 我设成了35
他代码里设置的是 24
会不会和这个有关系

这样吧，infer一下把最大长度从35改到25
emse_2023-01-11-14-29-49
针对这个none的模型
CUDA_VISIBLE_DEVICES=6 python main.py --dataset=emse --type=none --train=False --load_checkpoint=True
--checkpoint=emse_2023-01-11-14-29-49_best.pth
load不了，不知道为啥。。。

emse_2023-01-12-06-09-17
不行 load不出来东西

那这样吧，直接跑一下这个模型，然后把target调小
CUDA_VISIBLE_DEVICES=7 python main.py --dataset=emse --type=none --max_target_len=25
emse_2023-01-12-06-36-09
每次跑的结果都不一样 这咋回事？

###################
一个问题完全解决不了，就是重新load之后，发现根本infer不出来一样的结果
不知道是什么原因

ans:完美解决了，就是因为创建词表的时候用了set而不是list，所以导致每次创建出的词表可能不是一样的，从而使得结果对不上
所以这也就意味着，之前训练的都不可能load之后拿到结果正确的结果了，结果绝对是错的，所有这些检查点都作废了
但是这些模型应该都是没问题的。

##############
接下来需要干的事情，就是研究一下为什么高出这么多，这是第一个问题。
emse_2023-01-12-06-36-09
CUDA_VISIBLE_DEVICES=7 python main.py --dataset=emse --type=none --max_target_len=25
这个模型是设置了target长度和论文一致，
目的是研究为什么结果高出这么多
他需要和targetlen长度为35的进行比较，观察是否是因为这个的原因导致变化
emse_2023-01-11-14-29-49
目前来看，没有什么太大的差距，两个模型的bleu值基本上是完全一致的，大概只有0.2个点的差距
所以我觉得不会是这个原因导致的。
所以有没有可能是代码写错了呢？也就是计算指标的时候算错了，比方说是否里边有pad，有eos等，导致结果不一致？
目前通过infer结果来看，我计算时所用的语料都是没有特殊符号的。
那么论文代码的实现呢？

稍微有些不一样，他这里边每个token都low了，其他应该都是一致的

ok，整理清楚了，他因为是先创建的dataloader，然后用dataloader创建的词表
然后在创建dataloader的时候所有的summary单词都lower，
所以如果我要follow的话，就需要预处理阶段做好这个事情就可以了
草，完全一样，根本不是这个原因，

##############
我觉得可能是因为这个原因，就是指针网络，对于leafmask的这个操作起了作用
做个实验看一下是不是这个原因

上什么实验呢？
我觉得可以重新跑一下，none的模型，这个target长度部分，我觉得还是设置成25吧
注意！！：现在默认的target length是25

CUDA_VISIBLE_DEVICES=6 python main.py --dataset=emse --type=none
emse_2023-01-12-10-01-00

CUDA_VISIBLE_DEVICES=7 python main.py --dataset=emse --type=none --pointer_leaf_mask=False
emse_2023-01-12-10-00-27 可以kill

首先得有一个模型是，只要tree的，认为现在T5还没定，只要tree的模型是有意义的
CUDA_VISIBLE_DEVICES=2 python main.py --dataset=emse --type=tree
emse_2023-01-12-10-04-55

CUDA_VISIBLE_DEVICES=3 python main.py --dataset=emse --type=tree --pointer_leaf_mask=False
emse_2023-01-12-10-05-10 可以kill

然后这两个实验都是要有的，一个用来当作emnlp文章的baseline，另一个用来当作消融实验用

然后需要做的是对于T5的改进实验，现在还是得搞
有什么更好的位置编码吗？
###################################################
#############
pointer_leaf_mask 没有意义，不用跑了，原因不是因为这个
CUDA_VISIBLE_DEVICES=6 python main.py --dataset=emse --type=none
emse_2023-01-12-10-01-00
这个没跑完不知为何炸了
emse_2023-01-13-03-07-01

CUDA_VISIBLE_DEVICES=2 python main.py --dataset=emse --type=tree
emse_2023-01-12-10-04-55

CUDA_VISIBLE_DEVICES=7 python main.py --dataset=emse --type=sentinel --leaf_PE_Type=T5
emse_2023-01-12-13-17-52

为leaf设计新型的位置编码

####

看了下模型和之前的对比，基本上也没有差距
区别1，paformer在decoder部分用的是postnorm
encoder部分没有必要一定要保持一致，
其他的不同，初始化，embedding部分的乘以根号d
就是这三个的不同，

###########################################
初始化还有embedding部分我觉得没必要非得调的一样吧，这种确实也没办法
算了吧，我试一下把decoder的norm部分改成postnorm，然后看一下结果
如果结果还是很高，那就不能怪我了，这个问题我就不管了，只解决消融实验的问题了。
CUDA_VISIBLE_DEVICES=3 python main.py --dataset=emse --type=none --decoder_pre_norm=False
emse_2023-01-12-14-37-36
-》》》结论：果然是postnorm的问题，使用postnorm之后
在23轮的时候的结果是###Best Result At EP23, best_bleu=37.11223568899161
而在其他对比的时候，由于prenorm的方法跑到13轮就掉了，
所以只比13轮的结果，发现postnorm的效果从35降到了31.50
###Best Result At EP13, best_bleu=31.50292082499219
###Best Result At EP13, best_bleu=34.99407180354686
所以说明，我的结果相比于论文结果过高的原因就在于这个部分

按照之前对emse的实验结果可以发现，是从40.62降低到了约37左右
###Best Result At EP29, best_bleu=40.62549158300682
这个结果跟nerucodesum差不多，可以，这个结果是可以用的。

所以这个emse_2023-01-12-14-37-36的decoder_pre_norm=False是可以作为默认的实验设置了

#####################################################
开始研究哨兵以及更好的叶子位置编码

###############
CUDA_VISIBLE_DEVICES=0 python main.py --dataset=emse --type=sentinel --rot=True
emse_2023-01-12-14-42-40
这个可以和T5的哨兵版本进行对比
似乎是有一点提升的。

##############

###################
突然发现对leafmask on pointer的实验有问题，在forward的时候没有进行这个操作，所以只是在infer的时候用了
所以这部分需要重新实验
CUDA_VISIBLE_DEVICES=4 python main.py --dataset=emse --type=none --pointer_leaf_mask=False
emse_2023-01-12-14-39-40 与none进行对比即可
这个结果需要等一下 看看然后关机
不过细想一下，infer和train都不一致的情况下，两者仍然差别不大，那我感觉其实已经说明没什么作用了已经
目前来看确实是没什么差别存在。
--》结论：没什么大的差别，原因应该不在这
###Best Result At EP24, best_bleu=40.57389460992093
结束
##########################

吃完饭然后上几个实验，然后再稍微思考一下哨兵的设计，看有没有
能再挖掘的点，其实现在的状况感觉估计会不错的

如果不行的话，那就直接上decoderprenorm的结果得了，其实也不能算是作弊，毕竟prenorm和postnorm的选择都是可行的。
你现在充其量只能说在当前的超参数下，我的模型prenorm比postnorm，但是有可能他的模型再加上他的超参反而postnorm好，
都说不定。
