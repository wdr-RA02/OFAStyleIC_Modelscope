# OFAStyleIC_Modelscope
对Modelscope中OFA image caption的一系列组件进行适当修改，使得任务包含风格属性

## Updates
- 2023-03-19: finetune.py和pre_inference.py已被重构成models.{finetuner, evaluator, inference_pipeline}三个模块, 并由model_operator.py统一调用
- 2023-03-22: 实验性地加入scst功能(详见func/add_scst分支)

## How to use
### Train:
Simple! 只需要在terminal中输入
```sh
CUDA_VISIBLE_DEVICES=x python3 model_operator.py train --conf path/to/conf.json
```
然后坐等训完, 训练好的模型会存放在{work_dir}/output里面

### trainer_conf.json结构
```py
{
    "model_name":"要finetune的模型名称",
    "model_revision": "模型版本",
    "tokenize_style": true/false,          # 是否对风格进行tokenize处理
    "img_addr":"path_to_PIC/yfcc_images/", # 务必记得加最后那个/
    "dataset_path":"path_to_PIC/personality_captions/",
    "file_attr":".jpg",
    "train_json":"train.json",
    "test_json":"test.json",
    "val_json":"val.json",
    "work_dir":"path_to_work_dir/",
    "prompt":"" #目前还没实现这个功能
}
```
我提供了几个配置例子:
- [base_tokenized_pt](conf_examples/base_tokenized_pt.json)
- [tiny_tokenized_pt](conf_examples/tiny_tokenized_pt.json)
- [distilled_tokenized_coco](conf_examples/distilled_tokenized_coco.json)

### Inference:
Also simple! 
```sh
CUDA_VISIBLE_DEVICES=x python3 model_operator.py inference --conf path/to/conf.json
```

推理结果保存了一份在{work_dir}/inference_{time}.json里面, 文件结构:
```python
{
    "model_name": "damo/ofa_pretrain_tiny_en",  # base模型名字
    "model_revision": "v1.0.1",
    "n_eval_sample": 10,                        # 样本个数
    "results": [                     # list, 包含每个样本的reference cap和生成cap
        {
            "style":"Style",
            "image":"path/to/img",
            "caption":"xxx",
            "reference":"yyy",
        },...
    ]
}
```

## Model_operator.py arguments
| args                         | help                       | default          | available in          |
|------------------------------|----------------------------|------------------|-----------------------|
| -c/--conf path/to/train_conf | 指定train configuration json | *required param* | ALL                   |
| -b/--batch_size N            | batch大小                    | 4                | ALL                   |
| -p/--patch_image_size P      | resnet patch大小             | 224              | ALL                   |
| -m/--max_image_size M        | resnet max image大小         | 256              | ALL                   |
| -e/--max_epoches N           | 最多训练多少epoch                | 3                | ``train``             |
| -t/--checkpoint path/to/ckpt | 指定ckpt目录                   | None             | ``train``             |
| -w/--num workers N           | dataloader worker个数        | 0                | ``train`` && ``eval`` |


我目前准备使用的模型:
- damo/ofa_image-caption_coco_distilled_en, v1.0.1  [Modelscope](https://modelscope.cn/models/damo/ofa_image-caption_coco_distilled_en/summary)  |  [conf json](conf_examples/distilled_tokenized.json)
- damo/ofa_pretrain_base_en, v1.0.2  [Modelscope](https://modelscope.cn/models/damo/ofa_pretrain_base_en/summary)  |  [conf json](conf/base_tokenized.json)


## How to save model?
shell在代码根目录下输入: 
```sh
python3 -m utils.backup_model --conf path_to_conf
```
然后就可以在work_dir文件夹下找到包含模型和配置的tar.gz文件

## utils.backup_model arguments
| args                         | help                         | default          |
|------------------------------|------------------------------|------------------|
| -c/--conf path/to/train_conf | 指定train configuration json   | *required param* |
| -o/--out_dir                 | 指定输出目录                       | ./work_dir       |
| -e/--example_json            | 若使用该参数, 则会保存抹去dataset位置的json | False            |
| -n/--no_json                 | 若使用该参数, 则打包时会忽略配置json文件      | False            |

> 注意: -e/-n只能二择其一


## Credits
- [Personality_Captions](https://openaccess.thecvf.com/content_CVPR_2019/html/Shuster_Engaging_Image_Captioning_via_Personality_CVPR_2019_paper.html)

- [Modelscope](https://modelscope.cn)

- [OFA](https://github.com/OFA-Sys/OFA)

- pycocoevalcap: [1](https://github.com/salaniz/pycocoevalcap)  &  [2](https://github.com/tylin/coco-caption)
