## Problem log

> 2022-03-06
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


> 2022-03-09
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


> 2022-03-11

1. 昨天好歹是成功让服务器admin升级了一下驱动, 不然没法装modelscope...

虽然但是, 有点想写一个基于hf的fallback了. 毕竟说不好我就有可能得用另一台服务器, 那个只能装1.7的pytorch...

环境, 环境还是环境!!!

可以考虑一下开个分支搞一下tokenizer了

2. 换分支了, func/(origin/tokenized)_style. 把一些数据处理的函数剥离出来放到了utils里面.

可恶啊数据集可能没下好报了truncated image, 然后换huge又爆显存, 哎哎真是的

3. 周一要跟导师汇报希望还能留下全尸捏

搞完blip以后一定要用huggingface重构, modelscope的文档一言难尽...

4. 实在不行用base的pretrain ckpt弄吧