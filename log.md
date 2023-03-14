## Problem log

> Notes:
1. 一些缩写:
- OFAS(Style)ICP: OFAStylishICPreprocessor
- OFAPpSIC: OFAPreprocessorForStylishIC, 包含前者
- 


> 2023-03-06
### 1. prompt所在的位置

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


> 2023-03-09
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

鉴定为得在trainer里面指定preprocessor

电脑太烂显存只有4g 明天再试吧


> 2023-03-11

1. 昨天好歹是成功让服务器admin升级了一下驱动, 不然没法装modelscope...

虽然但是, 有点想写一个基于hf的fallback了. 毕竟说不好我就有可能得用另一台服务器, 那个只能装1.7的pytorch...

环境, 环境还是环境!!!

可以考虑一下开个分支搞一下tokenizer了

2. 换分支了, func/(origin/tokenized)_style. 把一些数据处理的函数剥离出来放到了utils里面.

可恶啊数据集可能没下好报了truncated image, 然后换huge又爆显存, 哎哎真是的

3. 周一要跟导师汇报希望还能留下全尸捏

搞完blip以后一定要用huggingface重构, modelscope的文档一言难尽...

4. 实在不行用base的pretrain ckpt弄吧

> 2023-03-12

太幽默了, 写了一早上处理数据集的函数 结果好不容易处理好了服务器cuda倒先崩了啊哈哈哈哈

加checkpoint教程: https://modelscope.cn/docs/Configuration%E8%AF%A6%E8%A7%A3

> 2023-03-13

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

> 2023-03-14
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
|--get_style_dict(style_list)->Dict[str, str]: 接受上述的list进行enumerate, 返回格式为{_STYLE: "<style_k>"}
|
|-finetune.py
|--generate_preprocessors(train_conf, work_dir, tokenized: bool)->Dict[str, OFAPpSIC]: 生成preprocessor, 基于是否tokenizer的情况进行分别处理
|--__main__
|--args:{--trainer_conf:train_conf.json位置,
         --checkpoint: 是否加载checkpoint
         mode: [train|eval]: 训练格式}

```

感觉origin_style分支可以就此落幕了

2. TODO: 
- **添加metric**
按照baseline的配置的话, 最起码包括B1/B4/rouge/CIDEr/spice

ref: https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%AF%84%E4%BC%B0

- 调查prompt的作用
- 调个base的模型试一下