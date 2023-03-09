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


