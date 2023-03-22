## Problem log

### Notes:
1. 一些缩写:
- OFAS(Style)ICP: OFAStylishICPreprocessor
- OFAPpSIC: OFAPreprocessorForStylishIC, 包含前者
- 


#### 2023-03-06
1. prompt所在的位置

```
|-OFAICPreprocessor(..., cfg, ...)
|---__call__(data: Dict) -> Dict #调用
|---_build_infer_sample(...):
        src=self.cfg.get("prompt", " what does the img describe")
```

该函数返回值为
```python
{"input": tokenized instructions,
 "patch_img": 图像特征,
 "mask": 遮罩,
 "label": caption tokenized
}
```

代表OFA caption需要的prompt在Preprocessor里。因此需要
- build一个OFAICPreprocessor的子类
> 根据pipeline提示, 在此基础上需要对OfaPreprocessor稍加修改
> 具体需要将其中的self.preprocess定死为OFAICPreprocessor
- 重写该方法修改prompt为"what does the image describe? write a {} reply.", 加special token
> *personality个数及插入方式?*


#### 2023-03-09
基于OfaPreprocessor实现OfaPreprocessorforStylishIC, 调用OfaStyleICP对数据集预处理

dataset remapping:
```python
{"personality":"style",
"comment":"text",
"image_hash":"image"}
```

遇到了很艹蛋的问题: trainer没法指定使用我的preprocessor!

modelscope官网给的解决方案是dataset转成huggingface那套然后再map

可是这样很占内存啊, 而且`build_trainer`之后他还是照例给我指派了preprocessor....

~~鉴定为不得不重写OFATrainer~~

鉴定为得在trainer里面指定preprocessor^[from 03-15: 这方法没完全起作用! T^T]


电脑太烂显存只有4g 明天再试吧


#### 2023-03-11

1. 昨天好歹是成功让服务器admin升级了一下驱动, 不然没法装modelscope...

虽然但是, 有点想写一个基于hf的fallback了. 毕竟说不好我就有可能得用另一台服务器, 那个只能装1.7的pytorch...

环境, 环境还是环境!!!

可以考虑一下开个分支搞一下tokenizer了

2. 换分支了, func/(origin/tokenized)_style. 把一些数据处理的函数剥离出来放到了utils里面.

可恶啊数据集可能没下好报了truncated image, 然后换huge又爆显存, 哎哎真是的

3. 周一要跟导师汇报希望还能留下全尸捏

搞完BLIP以后一定要用huggingface重构, modelscope的文档一言难尽...

4. 实在不行用base的pretrain ckpt弄吧

#### 2023-03-12

太幽默了, 写了一早上处理数据集的函数 结果好不容易处理好了服务器cuda倒先崩了啊哈哈哈哈

加checkpoint教程: https://modelscope.cn/docs/Configuration%E8%AF%A6%E8%A7%A3

#### 2023-03-13

这两天代码量比较少, 主要跑训练去了

checkpoint是通过checkpointhook构造的

```py
cfg.train.hooks=[{
        'type': 'CheckpointHook',
        'by_epoch': False,      #指定是按epoch保存还是按iter保存
        'interval': 5000,       #每隔多少次保存一下
        'max_checkpoint_num': 3 #最多保存几份, 后面的会挤掉前面的
    }, ...
]
```

最后会保存在work_dir文件夹下

#### 2023-03-14
1. 在func/tokenized_style分支里面添加了利用tokenized_style训练的函数

具体包括:
```python
|-preprocessors
|--sic.py
|---OFAPpSIC
|----tokenize_style: bool, 指示是否使用tokenize training 
|---OFASICP
|----tokenize_style: bool, 指示是否使用tokenize training 
|----add_style_token(self, style_dict)->None 将生成的style字典添加进preprocessor
|----_build_infer_sample(...)->dict: 进行了有关prompt的修改
|
|-utils
|--list_styles(ds_path, style_f_n)->list: 从personalities.txt文件中读取style并返回列表
|--get_style_dict(style_list)->Dict[str, str]: 
|            接受上述的list进行enumerate, 返回格式为{_STYLE: "<code_k>"}
|
|-finetune.py
|--generate_preprocessors(train_conf, 
|                         work_dir, 
|                         tokenized: bool)->Dict[str, OFAPpSIC]: 
|            生成preprocessor, 基于是否tokenizer的情况进行分别处理
|--__main__
|--args:{--conf:train_conf.json位置,
         --checkpoint: 是否加载checkpoint
         --max_epoches
         --batch_size
         --num_workers: 模型训练参数
         mode: [train|eval]: 训练格式}

```
> modified in 03-15: 
> - code_k<-style_k
> - add args


感觉origin_style分支可以就此落幕了

2. TODO: 
- **添加metric**
按照baseline的配置的话, 最起码包括B1/B4/rouge/CIDEr/spice

ref: https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%AF%84%E4%BC%B0

- 调查prompt的作用

- 调个base的模型试一下

#### 2023-03-15

1. 一点心声:

说实话蛮难过, 因为之前提及的添加``trainer.preprocessor``的方法没起作用, 训练实际上还是在用他们自己写的OfaPp. (实际上有用的方法是在args里面指定)

而且由于vocab_size是在OFAModel里面指定的, 如果自己加token需要自己多写几个类. 
为了不伤筋动骨不得不决定废物利用一下, 把style token改为<code_i>

今天GPT-4发布了, 好像是添加了多模态的处理, 感觉它的出现应该会把整个游戏杀死. 哎.

2. 托了上面那个发现的福, 在finetune那里把trainer的构造拆分了出来, 还发现了pcap里有几个不存在的personality. 能不能去提个issue捏

3. TODO: 同昨日

#### 2023-03-16

0. 
**永远先备份模型再删除!**

**永远先备份模型再删除!!**

**永远先备份模型再删除!!!**

你永远也不知道究竟是代码错了还是你错了!!!

好蠢好蠢, 居然把inference写错了怪罪到Preprocessor上去, 八小时心血白费了

具体而言, pre-inference那里我把风格全部小写了, 然后token不就不知道是怎么回事嘛
然后我就以为是Pp的问题就删了, 结果并不是然后之前那个就白训了^[modified in 2023-03-17]

1. 很好很好, 终于获得了像点样子的模型. 俗话说行百里者半九十不过就是这么回事罢

   本来今天的汇报又delay到了明天啊哈哈哈

2. TODO: 还是写metric, 然后训练多几个版本, 然后考虑一下传到modelscope里面

#### 2023-03-17

1. notes
metric原始输入:
```python
Input={
       'nsentences': n,
       'net_input': {'input_ids':    prompt: List[int], 
                     'patch_images': imgs: List[Tensor],
                     'labels':       [[input_captions_1], ...]: List[List[str]]
                    }, 
       'labels': one_data["text"]: List[List[str]] 
}

output={
    'caption': [[gen_caption_i], ...]: List[List[str]]    #每个List[str]里只有一个内容
    'samples': [{
        'style': style:str,
        'text':  input_cap:str,
        'image': img_addr:str
    },...]: List[Dict[str, str]]
}
```
this^[modified in 2023-03-19]

2. 
发现了一个大问题: 其他人训练都是cut成224\*224的, 然而我还搁着搞480\*480, 感觉应该有一点点关系, 不然BLEU怎么可能比Updown这些还差呢
   
晚上等base弄好了看看情况重训一个吧(熟悉的展开hh)

这两天该忙外文翻译了, 其实这种活用deepl也不是不能解


### 2023-03-18
改了image指标还是不行, 现在有几条路:
- 多几个epoch
- 改指令
- 看看encoder有没有冻结, 如果有就冻结它
- 使用OFAmain代码加CIDEr预训练任务
- 十分恐怖: 换其他框架

不过我认为这么强的模型, 有这么厉害的zero shot能力肯定还是我没设计好任务导致的

### 2023-03-19

1. 用pycocoeval重写了一下metric发现也没有那么差, 最起码比19年的baseline厉害点

顺手记录一下这玩意需要的输入输出

```python
# 模型的输出和ground truth都是这个格式
reference={
    "id_0":[{"caption": caption_0_j},...],
    "id_1":[{"caption": caption_1_j},...],
    ...
}

# 经过PTBTokenizer以后
reference_ptb={
    "id_0":[caption_0_0, caption_0_1,...],
    ...
}
```

2. TODO: 看一下怎么提升性能了, 然后慢慢把一些东西迁移到hf那套去, 只留绝对必要的东西在modelscope下面

### 2023-03-20
1. 本来因为能在preprocessor里面去掉image tensor的, 结果一去性能炸裂, 我也不知道怎么回事, 只好先加上了

2. 我抄, 有个OFAsys的库, 让我有点感觉像小丑


### 2023-03-21
参考了一下OFAmain的SCST实现, 最后感觉有两条路子走:

- 1. Updown方案: 
  - gen:=ramdom.choice(beam_decoded)
  - ground_truth=PIC.train_set (halve)
  - baseline:=tokenizer.batch_decode(softmax(logits, dim=-1))

- 2. OFA方案:
  - gen:=model.inference()
  - ground_truth=PIC.test_set (halve)
  - baseline:=CIDEr_D(me, [other_gts_in_the_batch])

然后还发现我一直找不到的OFAmain: self.task.scst_generator竟然在modelscope.models.multi_modal.ofa.generate.SeqGenerator里面, 并且OfaModelForAllTasks._txt_gen_inference也调用了

gen_out=generate([model], sample):={
    [
        [
            {
                "tokens": Tensor,
                "score": Tensor(int),
                "attention": Tensor,
                "alignment": Tensor
            }
        ]
    ]
}

当然我不知道scst_generator是不是还有别的玄机哈, 总而言之我先复现一个updown版本的试试看

我只有一个小小的愿望, 那就是打过GPTSpeaker就行了

### 2023-03-22

1. 大无语了, 本来想着一个batch里一个个算cider, 结果每个都是0. 难崩.

后来发现要成一个batch输进去才行, 果然道行还是不够深T^T

2. 关于OFAmain中计算scst_loss的一条代码:

```python
loss=-lprobs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze() ...

```

解释:
lprob.shape=[b, num_words, vocab_size]
target.shape=[b, num_words]
这一条的作用是在lprob中找出target[句子][第i个单词]对应词的概率lprob[句子][第i个单词][target[句子][第i个单词]]


