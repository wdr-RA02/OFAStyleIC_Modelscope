## Problem log

### 1. prompt所在的位置

```
|-OFAICPreprocessor(..., cfg, ...)
|---__call__(data: Dict) -> Dict #调用
|---_build_infer_sample(...):
        src=self.cfg.get("prompt", " what does the img describe")
```

该函数返回值为
```python
{"input": 图像特征,
 "patch_img": 图像特征,
 "mask": 遮罩,
 "label": caption tokenized
}
```

代表OFA caption需要的prompt在Preprocessor里。因此需要
- build一个OFAICPreprocessor的子类
> 根据pipeline提示, 在此基础上需要对OfaPreprocessor稍加修改
> 具体需要将其中的self.preprocess定死为OFAICPreprocessor
- 重写该方法修改prompt为"Describe the image in style <>", 加special token
> *personality个数及插入方式?*
- 

### 2. finetune时是否加入其他loss做目标
